# Math_exercise_1

2026 数学建模训练题 B：海上油田人员直升机运载计划编排。

当前阶段：**Phase 1 — Infrastructure MVP**。

本阶段只建立可验证的 Q1 基础设施，不实现优化算法。目标是让一个手工构造的 Q1 `Solution` 能够完成：

1. 读取题目数据；
2. 内部数据结构表示；
3. Fast Checker 可行性检查；
4. Evaluator 指标计算；
5. 官方 CSV 导出；
6. 独立 Reference Validator 重新读取 CSV 并复算；
7. GitHub Actions 自动测试。

优化算法将在 Phase 2 通过统一 `Solver` 接口接入。
