from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import AIRPORTS, Assignment, FlightPlan, SeaStop, Solution
from helicopter_planner.evaluation.metrics import evaluate_solution, leg_minutes
from helicopter_planner.io.load_problem import load_q2_problem
from helicopter_planner.io.load_solution_csv import load_exported_solution
from helicopter_planner.solver.q2.tail_elimination import Q2PassengerSetRouteOptimizer
from reference_validator import validate_submission

TARGET_KEYS = (("T2", 25), ("T2", 26), ("T3", 41), ("T3", 57))
U, V = "F044", "F037"


@dataclass(frozen=True)
class RepeatRoute:
    base: str
    aircraft_type: str
    flags: tuple[bool, bool, bool]
    trip_minutes: int
    distance_km: float


@dataclass(frozen=True)
class SourceOption:
    loads: tuple[int, int, int, int]
    moved_count: int
    residual_minutes: int
    moved_people: tuple[str, ...]
    residual_people: tuple[str, ...]


def person_ids(solution, uid):
    return sorted(pid for pid, a in solution.assignments.items() if a.flight_uid == uid)


def find_uid(solution, aircraft_type, flight_no):
    suffix = f"-{aircraft_type}-{flight_no:04d}"
    hits = [u for u in solution.flights if u.endswith(suffix)]
    if len(hits) != 1:
        raise RuntimeError((aircraft_type, flight_no, hits))
    return hits[0]


def flight_minutes(problem, solution, uid):
    aa = {p:a for p,a in solution.assignments.items() if a.flight_uid == uid}
    return evaluate_solution(problem, Solution(flights={uid:solution.flights[uid]}, assignments=aa)).total_aircraft_usage_minutes


def interval(origin, destination, base):
    # A -> U(first) -> V -> U(second) -> A
    if origin == "LAND": o = 0
    elif origin in AIRPORTS:
        if origin != base: return None
        o = 0
    elif origin == U: o = 1 if destination == V else 3
    elif origin == V: o = 2
    else: return None

    if destination == "LAND": d = 4
    elif destination in AIRPORTS:
        if destination != base: return None
        d = 4
    elif destination == U: d = 3 if origin == V else 1
    elif destination == V: d = 2
    else: return None
    return (o,d) if o < d else None


def build_repeat(problem, base, aircraft_type):
    spec = problem.aircraft_types[aircraft_type]
    best = None
    for flags in ((False,False,False),(True,False,False),(False,False,True),(True,False,True)):
        fuel = spec.tank_capacity_kg
        elapsed = 0; dist = 0.0; cur = base; ok = True
        for i,nxt in enumerate((U,V,U)):
            d = problem.distance(cur,nxt); fuel -= d*spec.fuel_rate_kg_per_km
            if fuel + 1e-9 < spec.min_safe_fuel_kg: ok=False; break
            elapsed += leg_minutes(d,spec.speed_kmh); dist += d
            if flags[i]:
                if nxt != U: ok=False; break
                elapsed += 20; fuel = spec.tank_capacity_kg
            else: elapsed += 10
            cur=nxt
        if not ok: continue
        d=problem.distance(cur,base); fuel -= d*spec.fuel_rate_kg_per_km
        if fuel + 1e-9 < spec.min_safe_fuel_kg: continue
        elapsed += leg_minutes(d,spec.speed_kmh); dist += d
        key=(elapsed,dist,flags)
        rr=RepeatRoute(base,aircraft_type,flags,elapsed,dist)
        if best is None or key < best[0]: best=(key,rr)
    return None if best is None else best[1]


def enumerate_source_options(problem, optimizer, all_ids, base, capacity):
    # Group repeat-compatible requests by exact OD and resulting interval.
    groups=[]
    incompatible=[]
    by=defaultdict(list)
    for pid in all_ids:
        r=problem.requests[pid]; iv=interval(r.origin_id,r.destination_id,base)
        if iv is None: incompatible.append(pid)
        else: by[(r.origin_id,r.destination_id,iv)].append(pid)
    for key,ids in sorted(by.items()): groups.append((key,tuple(sorted(ids))))

    # Recursively enumerate count choices with incremental leg-capacity pruning.
    raw=[]
    counts=[0]*len(groups)
    loads=[0,0,0,0]
    def rec(i):
        if i==len(groups):
            moved=[]
            for j,(_,ids) in enumerate(groups): moved.extend(ids[:counts[j]])
            moved_set=set(moved)
            residual=tuple(sorted(pid for pid in all_ids if pid not in moved_set))
            if residual:
                opt=optimizer.optimize(problem, optimizer.profile(problem,residual))
                if opt is None: return
                residual_minutes=opt.aircraft_minutes
            else:
                residual_minutes=0
            raw.append(SourceOption(tuple(loads),len(moved),residual_minutes,tuple(sorted(moved)),residual))
            return
        (_,_,iv),ids=groups[i]
        a,b=iv
        maxn=len(ids)
        for n in range(maxn+1):
            if any(loads[l]+n>capacity for l in range(a,b)): break
            counts[i]=n
            for l in range(a,b): loads[l]+=n
            rec(i+1)
            for l in range(a,b): loads[l]-=n
        counts[i]=0
    rec(0)

    # Exact dominance for DP: same load vector -> minimum residual minutes;
    # on ties prefer more moved people and deterministic IDs.
    best={}
    for o in raw:
        key=o.loads
        rank=(o.residual_minutes,-o.moved_count,o.moved_people)
        if key not in best or rank < best[key][0]: best[key]=(rank,o)
    return tuple(v[1] for v in best.values()), len(raw)


def install_repeat(solution, uid, rr, pids, problem):
    solution.flights[uid]=FlightPlan(uid,rr.base,rr.aircraft_type,[SeaStop(U,rr.flags[0]),SeaStop(V,False),SeaStop(U,rr.flags[2])])
    for pid in pids:
        r=problem.requests[pid]; iv=interval(r.origin_id,r.destination_id,rr.base)
        if iv is None: raise AssertionError(pid)
        solution.assignments[pid]=Assignment(pid,uid,iv[0],iv[1])


def export(solution,routes_path,assign_path):
    by=defaultdict(list)
    for uid,f in solution.flights.items(): by[f.aircraft_type].append(uid)
    no={}
    for t in sorted(by):
        for i,uid in enumerate(sorted(by[t]),1): no[uid]=i
    with routes_path.open('w',newline='',encoding='utf-8') as h:
        w=csv.writer(h); w.writerow(['aircraft_type','flight_no','stop_order','facility_id','refuel'])
        for uid in sorted(solution.flights,key=lambda u:(solution.flights[u].aircraft_type,no[u])):
            f=solution.flights[uid]; n=no[uid]
            w.writerow([f.aircraft_type,n,0,f.base_airport,0])
            for i,s in enumerate(f.sea_stops,1): w.writerow([f.aircraft_type,n,i,s.facility_id,int(s.refuel)])
            w.writerow([f.aircraft_type,n,len(f.sea_stops)+1,f.base_airport,0])
    with assign_path.open('w',newline='',encoding='utf-8') as h:
        w=csv.writer(h); w.writerow(['person_id','aircraft_type','flight_no','pickup_stop_order','delivery_stop_order'])
        for pid in sorted(solution.assignments):
            a=solution.assignments[pid]; f=solution.flights[a.flight_uid]
            w.writerow([pid,f.aircraft_type,no[a.flight_uid],a.pickup_index,a.delivery_index])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--anchor-routes',type=Path,required=True); ap.add_argument('--anchor-assignments',type=Path,required=True); ap.add_argument('--output-dir',type=Path,default=Path('outputs/q2/c2b'))
    args=ap.parse_args(); out=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir; out.mkdir(parents=True,exist_ok=True)
    problem=load_q2_problem(ROOT/'data/raw/distances.csv',ROOT/'data/raw/peopleQ2.csv')
    anchor=load_exported_solution(args.anchor_routes,args.anchor_assignments,uid_prefix='q2-c2b-anchor')
    ck=check_solution(problem,anchor,require_all_requests=True)
    if not ck.ok: raise RuntimeError('\n'.join(ck.errors))
    am=evaluate_solution(problem,anchor)
    if am.total_aircraft_usage_minutes!=19126: raise RuntimeError(am.total_aircraft_usage_minutes)
    uids=[find_uid(anchor,*x) for x in TARGET_KEYS]
    ids={u:person_ids(anchor,u) for u in uids}
    old_region=sum(flight_minutes(problem,anchor,u) for u in uids)
    optimizer=Q2PassengerSetRouteOptimizer()

    best=None; diagnostics=[]
    for base in sorted(AIRPORTS):
        for typ in ('T1','T2','T3'):
            rr=build_repeat(problem,base,typ)
            if rr is None: continue
            cap=problem.aircraft_types[typ].seats
            source_opts=[]; raw_total=0
            for uid in uids:
                opts,raw=enumerate_source_options(problem,optimizer,ids[uid],base,cap)
                source_opts.append((uid,opts)); raw_total+=raw

            # DP state = 4 repeat-route leg loads. Value = (sum residual minutes, moved count, choices).
            dp={(0,0,0,0):(0,0,())}
            for uid,opts in source_opts:
                nd={}
                for state,(cost,moved,choices) in dp.items():
                    for o in opts:
                        ns=tuple(state[l]+o.loads[l] for l in range(4))
                        if any(x>cap for x in ns): continue
                        val=(cost+o.residual_minutes,moved+o.moved_count,choices+(o,))
                        rank=(val[0],-val[1],tuple(x.moved_people for x in val[2]))
                        prev=nd.get(ns)
                        if prev is None:
                            nd[ns]=val
                        else:
                            prank=(prev[0],-prev[1],tuple(x.moved_people for x in prev[2]))
                            if rank<prank: nd[ns]=val
                dp=nd
            config_best=None
            for loads,(rescost,moved,choices) in dp.items():
                if moved==0: continue
                total=rr.trip_minutes+rescost
                rank=(total,-moved,loads)
                if config_best is None or rank<config_best[0]: config_best=(rank,loads,rescost,moved,choices)
            if config_best is None: continue
            rank,loads,rescost,moved,choices=config_best
            regional=rr.trip_minutes+rescost
            diagnostics.append({'base':base,'aircraft_type':typ,'repeat_minutes':rr.trip_minutes,'moved':moved,'loads':list(loads),'residual_minutes':rescost,'regional_minutes':regional,'savings':old_region-regional,'raw_source_options':raw_total,'dp_states':len(dp)})
            gkey=(regional,-moved,typ,base)
            if best is None or gkey<best[0]: best=(gkey,rr,loads,choices)

    if best is None: raise RuntimeError('no C2b solution')
    _,rr,loads,choices=best
    result=Solution(flights=dict(anchor.flights),assignments=dict(anchor.assignments))
    moved=[]
    for uid,o in zip(uids,choices,strict=True):
        moved.extend(o.moved_people)
        if o.residual_people:
            opt=optimizer.optimize(problem,optimizer.profile(problem,o.residual_people))
            if opt is None: raise AssertionError(uid)
            optimizer.install(problem,result,uid,o.residual_people,opt)
        else:
            del result.flights[uid]
    install_repeat(result,'q2-c2b-repeat-F044-F037-F044',rr,tuple(sorted(moved)),problem)
    ck=check_solution(problem,result,require_all_requests=True)
    if not ck.ok: raise RuntimeError('checker:\n'+'\n'.join(ck.errors))
    rm=evaluate_solution(problem,result)
    routes=out/'q2-routes.csv'; assignments=out/'q2-assignments.csv'; export(result,routes,assignments)
    ref=validate_submission(ROOT/'data/raw/distances.csv',ROOT/'data/raw/peopleQ2.csv',routes,assignments,require_all_requests=True)
    if not ref.ok: raise RuntimeError('validator:\n'+'\n'.join(ref.errors))
    savings=am.total_aircraft_usage_minutes-rm.total_aircraft_usage_minutes
    summary={'stage':'Q2-C2b exact DP extraction for one F044-F037-F044 repeated route','anchor_minutes':19126,'old_region_minutes':old_region,'new_region_minutes':old_region-savings,'regional_savings_minutes':savings,'new_global_minutes':rm.total_aircraft_usage_minutes,'new_global_flights':rm.number_of_flights,'seat_utilization':rm.seat_utilization,'repeat_base':rr.base,'repeat_aircraft_type':rr.aircraft_type,'repeat_trip_minutes':rr.trip_minutes,'repeat_refuel_flags':list(rr.flags),'repeat_leg_loads':list(loads),'moved_people':len(moved),'reference_validator':'PASS','config_diagnostics':sorted(diagnostics,key=lambda x:x['regional_minutes'])}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'moved_people.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.writer(h); w.writerow(['person_id','origin_id','destination_id'])
        for pid in sorted(moved):
            r=problem.requests[pid]; w.writerow([pid,r.origin_id,r.destination_id])
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
