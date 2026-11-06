
# 代码审计总表（28 脚本，2026-10-23 完成度 28/28）

## A. 论文主链路（数字直接入稿）—— 10 个

| 脚本 | 状态 | 说明 |
|---|---|---|
| aci_wrap.py | ⚠️→✅ | P1: grouped 小组守卫（防御修复）；P2: floor 分位（全员一致，声明） |
| aci_wrap_seeds.py | ✅ | 未发现问题（三种子中位/range 正确） |
| family_select.py | 🔴→✅ | **P0: σ val 宽度泄漏 sig_t** → 修复（sig_v）；修复后 8/8 选择不变 |
| block_bootstrap.py | ⚠️ | 静态列标签 sigma_evt 实为 naive（已在论文表题声明 unweighted residual）；在线 evt 配置与选择器有一处不一致（影响 CI 宽度尾差，不动表已对齐工件） |
| tail_true_bins.py | 🔴→✅ | **P0: hist 用 val 数组按 test 下标切片 + 预热段入分母 + 平滑初始化混用** → v2 全修复重跑 |
| enbpi_faithful.py | 🔴→✅ | **P0: 双侧 α/1−α = 80% 名义** → α/2 修复（覆盖 0.893–0.947）；off-by-one 修复 |
| persistence_edge.py | ⚠️ | P2: level-2 不截断/对称化（参考值，非主张路径） |
| conformal_cqr.py | ✅ | 分数方向正确（组C核验） |
| cache_musig.py | ✅ | 分段正确（组C核验） |
| cache_cqr.py | ✅ | qlo/qhi 无混用（组C核验） |

## B. 活跃依赖（被 A import 或支撑复现）—— 3 个

| 脚本 | 状态 | 说明 |
|---|---|---|
| calibration.py | ✅ | collect() 被 cache 脚本使用（分段/gnd 矩阵逻辑核验正确）；评估部分为旧探索 |
| facts.py | ✅ | static 判定 startswith 已修复；聚合正确 |
| gate.py | ✅→⚠️ | 白名单动态化后有效，但仍属抽查型（非全文数字全量对账）——诚实声明 |

## C. 基线/参考（论文叙述引用其数字）—— 2 个

| 脚本 | 状态 | 说明 |
|---|---|---|
| enbpi.py | ⚠️→退役 | K=300 旧版（无校正）；**口径混用已消除**：论文全面切换至 faithful 版（比值 0.37×–1.72× 重算入稿） |
| conformal_persistence.py | ⚠️ | P2: 缺有限样本校正 + ρ 边位置加权失真——仅参考叙述（32.5@0.971），非主张 |

## D. 非论文链路（训练配置生成/早期探索/诊断）—— 13 个

gen_baseedge_configs / gen_beta_configs / gen_beta_cqr / gen_prob_configs（配置生成，训练用）；
gen_tables（表已手工精修替代）；reselect_by_crps / sigma_rescale / sweep_sigma（早期探索，被族选择替代）；
revision_experiments（panel 修复期中间产物）；summarize / bootstrap_ci（被块 bootstrap 替代）；
early_best_audit（诊断）；conformal_twlp + conformal_prob.csv（**LEGACY 已标注**）

**结论**：3 个原则性 bug（尾覆盖 hist 错位、EnbPI 80% 名义、σ 选择泄漏）全部修复并重跑；
1 个口径混用（EnbPI 旧/新两版数字混入论文）统一为忠实版。主表（Table I/II）链路无 bug。
修复后：选择分布不变、覆盖结论不变、变化的数字（尾覆盖区间/EnbPI 比值与覆盖叙述）已全部入稿。
