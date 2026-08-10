# EXPERIMENT_PLAN_REVISION_AFDB_SOURCE.md
## AFDB 单源多目标跨数据集房颤实验修订方案

> **文档定位**
>
> 本文件是现有实验计划体系的最新修订文件，用于指导后续 Codex 实施。
> 它不删除、不覆盖此前任何冻结结果，也不要求回改 Stage 1–5、Stage 5A、Stage 5B、Stage 5B+、Stage 5C、Stage 5D 的既有 artifact 与报告。
>
> 当前文档优先级最高。后续若与旧文档在 source-domain 角色、LTAFDB 处理规则或实验顺序上冲突，以本文件为准。
>
> Codex 后续应按以下顺序阅读：
>
> 1. `EXPERIMENT_PLAN.md`
> 2. `EXPERIMENT_PLAN_SUPPLEMENT.md`
> 3. `EXPERIMENT_PLAN_UPDATE_AFTER_STAGE5B.md`
> 4. `STAGE5A_REPORT.md`
> 5. `STAGE5B_REPORT.md`
> 6. `STAGE5B_PLUS_REPORT.md`
> 7. `STAGE5C_REPORT.md`
> 8. `STAGE5D_REPORT.md`
> 9. 本文件 `EXPERIMENT_PLAN_REVISION_AFDB_SOURCE.md`
> 10. 当前 `TODO.md`
>
> **从本文件开始，新的主实验框架固定为：**
>
> ```text
> AFDB (labeled source)
>          ↓
>      M_AFDB
>          ↓
> ┌────────┼──────────────┐
> ↓        ↓              ↓
> CPSC2021 LTAFDB-clean1h SHDB-AF
> ```
>
> 即：
>
> > **单一低资源标注源域 AFDB → 三个异构外部目标域**
>
> 这一设置作为后续论文主 cross-domain benchmark。

---

# 1. 本次修订的三个核心原因

## 1.1 LTAFDB 前 1 小时统一剔除

根据新的数据质量判断，LTAFDB 每条记录开始后的前 1 小时存在较严重、难以通过常规预处理彻底消除的噪声。

因此后续正式实验中统一定义：

```text
LTAFDB-clean1h-v1
```

规则：

```text
所有 LTAFDB 记录：
time < 3600 s 的样本全部剔除
```

该规则必须：

- 对所有患者统一应用；
- 不依赖 AF/non-AF 标签；
- 不根据某个患者个别质量动态调整；
- 在 window index 生成前完成；
- 产生新的数据版本、manifest 和 index hash。

旧版：

```text
LTAFDB-original
```

继续保留，仅用于历史实验解释。

---

## 1.2 AFDB 改为主 source domain

AFDB 后续不再主要作为 target，而改为：

```text
Primary labeled source domain
```

后续核心模型：

```text
M_AFDB
```

由 AFDB 训练获得。

主测试目标：

```text
AFDB -> CPSC2021
AFDB -> LTAFDB-clean1h
AFDB -> SHDB-AF
```

---

## 1.3 Primary benchmark 改为 single-source multi-target

不再将：

```text
CPSC -> SHDB
LTAF -> AFDB
```

视为最终主性能实验。

这些旧实验继续保留为：

```text
exploratory / development / mechanism experiments
```

新的主 benchmark 统一使用：

```text
同一个 source
同一个训练协议
同一个 source model
同一个 source disease direction
同一个 source threshold protocol
```

分别测试三个 target。

这样更直接回答：

> 一个低资源 source-domain 学到的疾病识别结构，能否迁移到多个不同外部数据库？

---

# 2. 旧结果处理原则

以下结果全部冻结，不删除、不覆盖：

```text
Stage 1–5
Stage 5A
Stage 5B
Stage 5B+
Stage 5C
Stage 5D
```

涉及旧 LTAFDB 的结果必须在后续文档中标记：

```text
LTAFDB-original / pre-revision
```

包括但不限于：

- Stage 3 `M_LTAF`
- Stage 4 LTAF prototype/direction
- LTAF -> AFDB
- `M_LTAF` reference-space geometry
- LTAF source thresholds
- LTAF-related Stage 5D mechanism results

这些结果仍可用于说明：

```text
发现过程
机制探索
研究动机形成
```

但不得与新的 `LTAFDB-clean1h-v1` final benchmark 结果混为同一版本。

---

# 3. 新实验总流程

更新后的主实验顺序：

```text
Revision R1 — LTAFDB-clean1h-v1
        ↓
Revision R2 — AFDB Source Protocol
        ↓
Revision R3 — M_AFDB Mechanism Revalidation
        ↓
New Decision Gate
        ↓
Stage 6A / Stage 6B
        ↓
AFDB -> 3 Targets Final Benchmark
        ↓
Patient-level Statistics
```

---

# 4. Revision R1 — LTAFDB Clean-1h

## 4.1 数据版本

新增：

```text
dataset version:
ltaf_skip_first_hour_v1
```

推荐输出：

```text
outputs/revision_r1_ltaf_clean1h/
```

以及新的：

```text
window index
dataset summary
manifest
SHA-256
```

## 4.2 切窗规则

原始统一规则继续保持：

```text
window = 10 s
stride = 10 s
target fs = 200 Hz
shape = [2, 2000]
strict AF / non-AF intervals only
transition windows excluded
```

新增限制：

```text
window_start_time >= 3600 s
```

如果窗口跨越 3600 s 边界：

```text
exclude
```

不允许截取半个旧窗口后补齐。

## 4.3 LTAF 数据审计

必须重新输出：

```text
subjects
records
AF windows
non-AF windows
excluded windows
removed first-hour windows
remaining duration
per-subject window count
```

并对比：

```text
LTAFDB-original
vs
LTAFDB-clean1h-v1
```

## 4.4 前 1 小时信号质量审计

该分析不作为模型创新，只作为数据处理依据。

目标：

> 给出客观证据说明前 1 小时质量总体更差。

建议比较：

```text
0–1 h
vs
>=1 h
```

至少统计以下 3–5 个简单、无标签质量指标：

- finite-value ratio
- flatline ratio
- clipping/extreme amplitude ratio
- extreme first-difference ratio
- high-frequency power ratio
- optional simple SQI

要求：

- 不使用 AF/non-AF 标签决定剔除；
- 不据此做患者特异性动态过滤；
- 只验证固定“前 1 小时删除”规则的合理性。

输出：

```text
ltaf_quality_before_after.csv
ltaf_quality_summary.json
ltaf_quality_comparison.png
```

## 4.5 R1 完成标准

必须确认：

- 新旧 LTAF index 可同时存在；
- 新 index hash 唯一；
- 无患者泄漏；
- 所有窗口 start time >= 3600 s；
- target loader 可正常读取；
- preprocessing 输出仍为 finite float32 `[2, 2000]`；
- 不修改旧 LTAF artifact。

---

# 5. Revision R2 — AFDB Source Training Protocol

## 5.1 关键问题

AFDB 只有少量有效 signal subjects，因此不继续采用简单：

```text
16 train / 3 val / 4 test
```

作为最终主 source protocol。

新的 source protocol 使用：

> **subject-level K-fold OOF development + full-source final training**

---

# 6. AFDB 5-fold Subject CV

## 6.1 Fold 原则

建议：

```text
K = 5
seed = 42
group = subject_id
```

严格禁止：

```text
window-level split
```

每位患者在一个 fold 中只能属于一个 partition。

## 6.2 每折训练

每折：

```text
train subjects
→ MedTS-TTT
→ validation subjects
```

继续使用现有 CE baseline 配置作为第一版：

```text
dim = 128
num_layers = 6
num_heads = 8
patch_size = 8
num_classes = 2
```

训练窗口仍沿用：

```text
max_windows_per_subject_per_class = 500
subject/class balanced sampler
```

## 6.3 每折输出

每 fold 保存：

```text
best checkpoint
best epoch
validation metrics
validation continuous scores
run manifest
dataset/index hash
seed
Git commit
```

---

# 7. AFDB OOF Source Predictions

## 7.1 目的

把五折 validation predictions 拼接，形成：

```text
AFDB OOF score archive
```

每个 AFDB subject 的预测必须来自：

> 没有见过该 subject 的模型。

## 7.2 OOF 用途

允许 OOF label 用于：

```text
source performance estimation
source threshold selection
source disease-score calibration
best epoch aggregation
source mechanism analysis
```

禁止 OOF 结果接触任何 target label。

## 7.3 AFDB strong source threshold

从 OOF continuous score 中选择：

```text
t_AFDB_head*
```

主规则：

```text
argmax Balanced Accuracy
```

tie rule 继承 Stage 5C。

同样为 prototype score 建立：

```text
t_AFDB_proto*
```

最终新的强 baseline：

```text
H1_AFDB = head score + OOF threshold
P1_AFDB = prototype score + OOF threshold
```

---

# 8. AFDB Final Full-Source Model

## 8.1 目的

五折开发完成后，用全部有效 AFDB source subjects 训练最终：

```text
M_AFDB
```

## 8.2 Final epoch

禁止使用 target 结果决定训练 epoch。

建议：

```text
final_epoch = median(best_epoch across 5 folds)
```

或使用预先冻结的其他聚合规则。

规则必须在 final full-source training 前写入 artifact。

## 8.3 Final training

使用：

```text
all eligible AFDB subjects
```

不再留出 target-like test patient。

因为最终泛化能力将在：

```text
CPSC2021
LTAFDB-clean1h
SHDB-AF
```

外部数据上评价。

---

# 9. AFDB 多 seed 稳定性

由于 source subjects 较少，至少运行：

```text
seed = 42
seed = 2024
seed = 3407
```

建议策略：

### Main analysis
```text
seed 42
```

### Stability
```text
2024
3407
```

最终至少报告：

```text
mean ± std
```

或：

```text
main seed + 2-seed sensitivity
```

禁止根据 target 表现挑 seed。

---

# 10. Revision R3 — M_AFDB Mechanism Revalidation

新 source 完成以后，必须在统一：

```text
M_AFDB
```

feature space 中重新验证核心机制。

不机械重跑所有历史 Stage，而只重跑下面四类必要分析。

---

# 11. R3-A — AFDB Prototype vs Linear Head

重新计算：

```text
d_proto_AFDB
d_head_AFDB
```

输出：

```text
cosine
angle
Pearson
Spearman
source OOF AUROC/AUPRC
target AUROC/AUPRC
```

目标：

判断 Stage 5A 的结论能否在 AFDB source 下复现：

> prototype disease direction 与 linear head 相关但不等价。

---

# 12. R3-B — Four-Dataset Disease Geometry in M_AFDB

在同一个冻结：

```text
M_AFDB
```

feature space 中提取：

```text
d_AFDB
d_CPSC
d_LTAF_clean
d_SHDB
```

构建：

```text
4 × 4 disease-direction cosine matrix
4 × 4 centroid-distance matrix
```

注意：

```text
LTAF 必须使用 LTAFDB-clean1h-v1
```

target labels 仅可用于：

```text
post-hoc mechanism analysis only
```

不得用于 adaptation。

---

# 13. R3-C — AFDB Head vs Shared Disease Axis

计算：

```text
d_shared = normalize(d_AFDB + d_CPSC + d_LTAF_clean + d_SHDB)
```

比较：

```text
cos(d_head_AFDB, d_AFDB)
cos(d_head_AFDB, d_CPSC)
cos(d_head_AFDB, d_LTAF_clean)
cos(d_head_AFDB, d_SHDB)
cos(d_head_AFDB, d_shared)
```

同时比较：

```text
mean prototype-prototype cosine
vs
mean head-to-prototype cosine
```

目标：

验证 Stage 5B+ 的结论是否在 AFDB source 上继续成立。

---

# 14. R3-D — AFDB Disease-Axis Distribution Shift

在：

```text
M_AFDB
```

中固定 AFDB source disease direction：

```text
d_source = d_AFDB
```

对三个 target：

```text
CPSC2021
LTAFDB-clean1h
SHDB-AF
```

计算：

```text
mu_nonAF
mu_AF
std_nonAF
std_AF
class gap
gap/source ratio
d-prime
overlap
AUROC
AUPRC
AF prevalence
source threshold
target oracle threshold
oracle BACC
P1 BACC
boundary headroom
```

target oracle：

```text
post-hoc mechanism only
```

不得进入 adaptation 配置。

---

# 15. 新 Decision Gate

R3-D 完成后重新决定 Stage 6 顺序。

旧 Stage 5D 的：

```text
Stage 6B first
```

结论不自动继承。

## Case A — Representation first

如果一个或多个主要 target 出现：

```text
AUROC < 0.90
或
AUPRC明显下降
或
gap/source ratio显著收缩
或
overlap明显升高
且
oracle threshold 只能小幅改善 BACC
```

则：

```text
Stage 6A first
```

优先改善 cross-domain representation/separation。

## Case B — Boundary first

如果三个 target 大多表现为：

```text
AUROC / AUPRC 较强
class separation仍良好
但
source threshold 与 target oracle threshold明显偏移
```

则：

```text
Stage 6B first
```

优先做 target boundary reconstruction。

## Case C — Mixed

如果：

```text
部分 target 是 representation 问题
部分 target 是 boundary 问题
```

则最终方法采用：

```text
representation enhancement
+
boundary reconstruction
```

但开发顺序优先解决对总体性能影响最大的瓶颈。

---

# 16. 新 Primary Benchmark

最终主结果固定为：

```text
AFDB -> CPSC2021
AFDB -> LTAFDB-clean1h
AFDB -> SHDB-AF
```

三个 target 必须：

- 使用同一个 `M_AFDB`；
- 使用同一个 source disease direction；
- 使用同一个 source threshold protocol；
- 使用同一个 adaptation algorithm；
- 使用同一套 frozen hyperparameters。

禁止：

```text
target-specific 手动调参
```

---

# 17. Target 角色划分建议

为减少隐式 target overfitting：

### Development target
```text
AFDB -> CPSC2021
```

允许：

```text
debug
method design
有限超参数选择
```

### Transfer-validation target
```text
AFDB -> LTAFDB-clean1h
```

用于：

```text
有限方法确认
```

### Final target
```text
AFDB -> SHDB-AF
```

尽量在方法冻结后用于最终外部验证。

若该 target 的历史结果已被多次查看，至少必须保证：

```text
Stage 6 超参数不根据 AFDB -> SHDB-AF 结果调整
```

---

# 18. 新 baseline 体系

最终至少保留：

## H0
```text
M_AFDB linear head
threshold = 0
```

等价于：

```text
P(AF) >= 0.5
```

## H1
```text
M_AFDB linear head
threshold = t_AFDB_head* from OOF
```

### 主 source baseline

## P0
```text
AFDB prototype score
source fixed prototype threshold
```

## P1
```text
AFDB prototype score
OOF optimized threshold
```

## GMM-v1
```text
existing unrestricted 2-component target GMM
```

## Ours
后续 Stage 6 确定的：

```text
stable disease-axis based cross-domain adaptation
```

---

# 19. Stage 6A — Representation Enhancement

仅在新的 Decision Gate 证明需要时进入。

第一版仍保留：

```text
R0: CE
R1: CE + SupCon
R2: CE + Prototype/Center
R3: CE + SupCon + Prototype
```

但新的主要评价不再是：

```text
direction cosine
```

因为方向可能已经高度稳定。

重点：

```text
target AUROC
target AUPRC
gap/source ratio
d-prime
overlap
```

目标：

> 保持疾病方向稳定的同时，提高跨域 class separation。

---

# 20. Stage 6B — Target Boundary Reconstruction

若新的 Decision Gate 支持 boundary-first，则优先开发。

核心候选：

### B6-1
```text
subject-balanced GMM
```

### B6-2
```text
source-anchored GMM
```

### B6-3
```text
constrained target-axis shift/scale model
```

例如：

```text
s_target ≈ a * s_source + b
a > 0
```

目标域无标签估计：

```text
a
b
pi_target
```

必要时再允许少量 variance adjustment。

### B6-4
```text
reliability fallback
```

若目标拟合不可靠：

```text
fallback -> P1 or H1
```

规则必须在 development target 冻结。

---

# 21. Boundary Headroom Recovery

Stage 6B 建议增加：

```text
Boundary Headroom Recovery
```

定义：

```text
(method_BACC - P1_BACC)
/
(oracle_BACC - P1_BACC)
```

仅在：

```text
oracle_BACC > P1_BACC
```

时解释。

作用：

> 衡量无标签 target adaptation 恢复了多少理论上可由 threshold correction 修复的性能空间。

该指标用于机制分析，不作为唯一主评价指标。

---

# 22. 最终主评价指标

## Ranking

```text
AUROC
AUPRC
```

## Operating point

```text
Balanced Accuracy
Macro-F1
MCC
Sensitivity
Specificity
Precision
Accuracy
```

论文主表建议突出：

```text
AUROC
AUPRC
Balanced Accuracy
Macro-F1
MCC
```

---

# 23. 统计方案

最终三个 target 必须使用：

```text
patient-level bootstrap
```

建议：

```text
1000 repetitions
```

报告：

```text
Baseline vs Ours
ΔAUROC
ΔAUPRC
ΔBACC
ΔMacro-F1
ΔMCC
95% CI
```

禁止 window-level bootstrap 代替患者级 bootstrap。

---

# 24. 推荐输出目录

```text
outputs/
├── revision_r1_ltaf_clean1h/
├── revision_r2_afdb_cv/
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   ├── fold_5/
│   ├── oof/
│   └── final_model/
├── revision_r3_afdb_mechanism/
│   ├── head_vs_direction/
│   ├── direction_geometry/
│   ├── head_vs_shared_axis/
│   └── axis_distribution_shift/
├── stage6a_representation_afdb/
├── stage6b_boundary_afdb/
├── final_afdb_multi_target/
└── final_statistics/
```

---

# 25. 推荐新增配置

```text
configs/datasets/ltaf_clean1h_v1.json
configs/experiments/afdb_cv_ce.json
configs/experiments/afdb_final_ce.json
configs/analysis/afdb_direction_geometry.json
configs/analysis/afdb_axis_shift.json
```

---

# 26. TODO 更新建议

```text
[x] Historical Stage 1–5
[x] Stage 5A
[x] Stage 5B
[x] Stage 5B+
[x] Stage 5C
[x] Stage 5D

[ ] Revision R1 — LTAFDB-clean1h-v1
    [ ] fixed first-hour removal
    [ ] quality audit
    [ ] rebuild index
    [ ] new dataset/version hash
    [ ] loader smoke
    [ ] report

[ ] Revision R2 — AFDB Source Protocol
    [ ] subject-level 5-fold split
    [ ] fold training
    [ ] OOF predictions
    [ ] OOF H1 threshold
    [ ] OOF P1 threshold
    [ ] final epoch rule
    [ ] full-source M_AFDB
    [ ] 3-seed stability
    [ ] report

[ ] Revision R3 — M_AFDB Mechanism
    [ ] R3-A prototype vs head
    [ ] R3-B four-dataset geometry
    [ ] R3-C head vs shared axis
    [ ] R3-D axis distribution / boundary shift
    [ ] new Decision Gate
    [ ] report

[ ] Stage 6A if justified
[ ] Stage 6B if justified

[ ] AFDB -> CPSC2021
[ ] AFDB -> LTAFDB-clean1h
[ ] AFDB -> SHDB-AF

[ ] patient bootstrap
[ ] final tables
[ ] final figures
```

---

# 27. Codex 工作纪律

后续 Codex 必须：

1. 不覆盖历史 Stage artifact；
2. 为 `LTAFDB-clean1h-v1` 创建独立 version；
3. 所有 AFDB fold 必须按 subject 划分；
4. 不允许 target label 参与：
   - source training；
   - source threshold；
   - model selection；
   - final epoch selection；
5. R3 机制分析中 target label 只能：
   ```text
   post-hoc mechanism analysis only
   ```
6. target oracle threshold 不得进入 Stage 6 配置；
7. final benchmark 三个 target 使用同一算法规则；
8. 不根据某一个 target 的好坏单独调整参数；
9. 每阶段先提交实施计划，再修改代码；
10. 每阶段完成后必须：
    - unit test；
    - real-data smoke；
    - formal run；
    - report；
    - TODO update；
    - clean Git commit。

---

# 28. Codex 当前第一优先任务

不要直接进入 Stage 6。

首先执行：

```text
Revision R1 — LTAFDB-clean1h-v1
```

第一步只做：

```text
代码与数据协议审计
```

Codex 应先：

1. 找到当前 LTAFDB window-index 构建位置；
2. 说明如何在不破坏旧 index 的情况下增加 `skip_first_seconds=3600`；
3. 列出预计新增/修改文件；
4. 定义新的 dataset/window version；
5. 设计 old-vs-clean 数据审计；
6. 设计 0–1h vs >=1h 简单无标签质量比较；
7. 明确输出 artifact；
8. 等待确认后再开始正式修改。

---

# 29. 当前项目新的核心研究命题

新的主实验希望最终回答：

> Can a disease-discriminative geometry learned from a small labeled AFDB source cohort remain useful across multiple heterogeneous ECG datasets, and can unlabeled target-domain adaptation recover domain-specific class separation and decision boundaries without target labels?

中文：

> **在低资源 AFDB 标注源域上学习得到的房颤疾病判别结构，能否迁移到多个异构 ECG 数据集，并在不使用目标域标签的前提下，通过目标域适应恢复跨域后的类别分离与决策边界？**

这将成为后续 Stage 6 与最终 benchmark 的统一实验目标。
