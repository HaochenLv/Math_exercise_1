from helicopter_planner.evaluation import leg_minutes

def test_official_example_leg_times():
    assert leg_minutes(235, 220) == 65
    assert leg_minutes(168, 220) == 46
    assert leg_minutes(171, 220) == 47
