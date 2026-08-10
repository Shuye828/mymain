# EXPERIMENT_PLAN_UPDATE_AFTER_STAGE5B.md
## Stage 5A/5B 结果驱动的后续实验更新计划

> **文档定位**
>
> 本文档是现有 `EXPERIMENT_PLAN.md` 与 `EXPERIMENT_PLAN_SUPPLEMENT.md` 的进一步更新文件。  
> 它不替代 Stage 1–5 已冻结的数据、代码和结果，也不要求回改既有报告。  
> **从 Stage 5B 完成以后，后续实验优先级、解释口径和 Stage 5C–Stage 6 的执行顺序以本文档为准。**
>
> Codex 在继续实验前应按以下顺序阅读：
>
> 1. `EXPERIMENT_PLAN.md`
> 2. `EXPERIMENT_PLAN_SUPPLEMENT.md`
> 3. `STAGE5A_REPORT.md`
> 4. `STAGE5B_REPORT.md`
> 5. 本文件 `EXPERIMENT_PLAN_UPDATE_AFTER_STAGE5B.md`
> 6. 当前 `TODO.md`

---

# 1. 当前已经确认的核心机制结论

## 1.1 Stage 5A：Prototype disease direction 不等价于 linear head

当前定义：

```text
d_proto = normalize(c_AF - c_nonAF)

d_head = normalize(w_AF - w_nonAF)

prototype_score = normalize(h) dot d_proto

classifier_score = logit_AF - logit_nonAF
```

正式结果：

| Source model | cosine(d_proto, d_head) | angle | minimum split Spearman |
|---|---:|---:|---:|
| CPSC2021 | 0.8553 | 31.21° | 0.7873 |
| LTAFDB | 0.9025 | 25.52° | 0.9459 |

因此：

- 两个方向具有明显共享判别成分；
- 但没有达到预注册的“高度等价”标准；
- prototype direction 不是 linear head 的简单重命名；
- 当前 `c_AF - c_nonAF` 仍不能单独作为充分创新点；
- 后续应研究其是否构成更稳定的跨域疾病几何结构。

跨域 ranking 结果也说明 prototype 并非普遍优于 head：

```text
CPSC2021 -> SHDB-AF
Prototype AUROC = 0.956959
Head      AUROC = 0.957772
Prototype AUPRC = 0.801630
Head      AUPRC = 0.786512

LTAFDB -> AFDB
Prototype AUROC = 0.898232
Head      AUROC = 0.899813
Prototype AUPRC = 0.876159
Head      AUPRC = 0.879709
```

因此当前正确表述是：

> Prototype direction 与 source linear head 相关但不等价，并表现出数据集相关的潜在跨域优势，而不是普遍优于分类头。

---

## 1.2 Stage 5B：四个数据集的 AF−non-AF 相对方向高度一致

在两个完全冻结、相互独立的参考特征空间中：

### `M_CPSC`

```text
跨数据集 disease-direction cosine:
range = 0.9443–0.9945
mean  = 0.9747

domain-centroid distance:
range = 0.1264–0.4541
```

### `M_LTAF`

```text
跨数据集 disease-direction cosine:
range = 0.9841–0.9929
mean  = 0.9892

domain-centroid distance:
range = 0.1631–0.3203
```

subject-equal sensitivity analysis 后：

```text
minimum disease-direction cosine = 0.9426
```

因此 Stage 5B 提供了强描述性证据：

> 四个 ECG 数据集在冻结特征空间中的绝对位置不同，但 AF 相对于 non-AF 的类别位移方向高度一致。

---

# 2. Stage 5A + 5B 合并后的新解释

现在项目的核心几何关系变为：

```text
不同数据库的绝对 feature location：
存在明显变化

不同数据库的 AF - nonAF relative direction：
高度一致

source linear head 与 prototype disease direction：
相关，但存在约 25–31° 偏离
```

因此后续论文主线不再是：

> “用 prototype direction 替代 classifier。”

而应调整为：

> Conventional source-domain classifiers learn decision axes optimized for the source distribution, whereas the AF-versus-non-AF relative prototype direction captures a highly conserved cross-dataset disease geometry. The remaining cross-domain error may therefore arise largely from changes in the class-conditional position, scale, overlap, and decision boundary along this stable disease axis.

对应中文：

> 源域分类头主要优化源域决策边界，而 AF−non-AF 原型方向体现了更稳定的跨数据库疾病相对几何结构。跨域后的主要问题可能不再是疾病方向发生大幅旋转，而是目标域在稳定疾病轴上的类别位置、尺度、重叠程度及分类边界发生变化。

---

# 3. 后续实验的核心问题重新排序

Stage 5B 后不应立即把主要精力放在继续提高 direction cosine，因为：

```text
M_CPSC mean cosine = 0.9747
M_LTAF mean cosine = 0.9892
```

方向一致性已经非常高，存在明显天花板效应。

后续优先回答：

### Q1
源分类头相对于“四数据库共享疾病方向”偏离多少？

### Q2
虽然 disease direction 稳定，但不同数据库沿这条轴的：

- non-AF mean
- AF mean
- class gap
- class variance
- overlap
- prevalence
- optimal boundary

是否发生系统性变化？

### Q3
source-validation threshold 是否能充分解决跨域边界漂移？

### Q4
无标签 target boundary reconstruction 能否稳定优于强 source threshold baseline？

### Q5
如果现有 disease direction 已经具有很强跨域 ranking，Stage 6 representation learning 是否仍能显著提升 AUROC/AUPRC，而不仅仅把 cosine 从 0.98 提高到 0.99？

---

# 4. 更新后的严格执行顺序

后续按以下顺序执行：

```text
Stage 5C   Strong Source Baseline
    ↓
Stage 5B+  Shared Disease Axis vs Source Linear Head
    ↓
Stage 5D   Disease-Axis Distribution & Boundary Shift Analysis
    ↓
Decision Gate
    ├── ranking / direction仍是主要瓶颈 → Stage 6A Representation Learning
    └── ranking已经稳定、boundary是主要瓶颈 → Stage 6B Boundary Reconstruction v2 优先
```

Stage 5C 必须先完成。

---

# 5. Stage 5C — Strong Source Baseline

## 5.1 目的

建立公平、足够强的 source-only baseline，避免最终方法只是击败一个未调阈值的 `0.5` baseline。

## 5.2 Linear-head baseline

### H0 — Original MedTS-TTT

```text
score = classifier probability / logit difference
threshold = 0.5 probability equivalent
```

保留 Stage 3/5 已有固定 0.5 结果。

### H1 — Strong source-threshold baseline

只在 source validation 上选择：

```text
t_head_source*
```

主规则冻结为：

```text
argmax Balanced Accuracy
```

然后：

```text
freeze threshold
→ apply unchanged to target
```

禁止 target label 参与阈值选择。

## 5.3 Prototype-direction baseline

同样建立：

### P0 — Existing fixed source prototype threshold

保留已有 source prototype midpoint / fixed threshold。

### P1 — Strong source-validation prototype threshold

在 source validation 上使用：

```text
prototype_score
```

选择：

```text
t_proto_source* = argmax Balanced Accuracy
```

之后完全冻结并应用 target。

## 5.4 Stage 5C 主比较

正式 target baseline 至少包含：

| ID | Score | Threshold |
|---|---|---|
| H0 | linear head | fixed 0.5 |
| H1 | linear head | source-val optimized |
| P0 | prototype direction | source fixed |
| P1 | prototype direction | source-val optimized |

其中论文的**主 source baseline**优先使用：

```text
H1
```

而目标边界重构必须至少同时挑战：

```text
H1
P1
```

## 5.5 指标解释

如果 score 不变而只改变 threshold：

```text
AUROC / AUPRC 必须保持完全一致
```

所以 Stage 5C 重点比较：

```text
Balanced Accuracy
Macro-F1
MCC
Sensitivity
Specificity
Precision
Accuracy
```

AUROC/AUPRC只用于确认ranking score没有发生变化。

## 5.6 Stage 5C 输出

建议：

```text
outputs/stage5c_strong_source_baseline/
├── cpsc2021/
├── ltafdb/
├── thresholds.csv
├── source_validation_curves.csv
├── target_results.csv
├── analysis_result.json
└── run_manifest.json
```

必须保存：

- threshold 来源；
- source validation metric；
- checkpoint hash；
- index hash；
- target label access audit；
- score hash。

---

# 6. Stage 5B+ — Shared Disease Axis vs Source Linear Head

## 6.1 目的

Stage 5A只比较：

```text
source prototype direction
vs
source linear-head direction
```

Stage 5B 已经获得同一 feature space 中四个数据集的真实 prototype directions。

因此现在直接补充：

```text
source linear head
vs
four-dataset prototype directions
```

这不需要重新训练模型。

## 6.2 必做分析

在 `M_CPSC` 空间：

```text
cos(d_head_CPSC, d_CPSC_proto)
cos(d_head_CPSC, d_LTAF_proto)
cos(d_head_CPSC, d_AFDB_proto)
cos(d_head_CPSC, d_SHDB_proto)
```

在 `M_LTAF` 空间：

```text
cos(d_head_LTAF, d_CPSC_proto)
cos(d_head_LTAF, d_LTAF_proto)
cos(d_head_LTAF, d_AFDB_proto)
cos(d_head_LTAF, d_SHDB_proto)
```

同时计算：

```text
mean prototype-prototype cosine
vs
mean head-to-prototype cosine
```

## 6.3 期待回答的问题

如果：

```text
prototype-prototype cosine ≈ 0.97–0.99

而

source-head to cross-dataset prototype cosine 明显更低
```

则支持：

> 四个数据集的 prototype directions 聚集在一个共享疾病轴附近，而 source-trained linear head 相对于该共享疾病几何发生了 source-specific rotation。

如果 head-to-prototype 同样接近 0.98–0.99，则不能声称 prototype 几何比 head 更跨域稳定，需要弱化该叙事。

## 6.4 输出

建议直接附加到：

```text
outputs/stage5b_direction_geometry/
```

新增：

```text
head_to_dataset_directions.csv
head_to_shared_axis_summary.json
head_vs_shared_axis.png
```

不覆盖原 Stage 5B artifact。

---

# 7. Stage 5D — Disease-Axis Distribution & Boundary Shift Analysis

## 7.1 这是 Stage 5B 后最关键的补充机制实验

当前已知：

```text
direction_A ≈ direction_B
```

接下来必须解释：

> 如果方向已经稳定，为什么跨数据集分类性能仍然下降？

核心假设：

> 目标域主要发生的是沿疾病轴的 class-conditional translation / scaling / overlap / prevalence / boundary shift，而不是疾病方向本身的大幅旋转。

## 7.2 投影定义

在每个冻结 source reference space 中，固定 source disease direction：

```text
d_source
```

对四个数据库所有统一选中的分析窗口：

```text
s = z dot d_source
```

必须使用与 Stage 5B 完全相同的 selected-window manifest，避免 cohort 改变。

## 7.3 每个数据集统计

利用真实标签，仅作：

```text
post-hoc mechanism analysis only
```

统计：

```text
mu_nonAF
mu_AF

std_nonAF
std_AF

class_gap = mu_AF - mu_nonAF

pooled_separation

AF prevalence

distribution overlap

source fixed threshold
source-val optimized threshold
target oracle threshold
```

其中 target oracle threshold：

- 只能用于机制分析；
- 不允许进入任何真实 adaptation；
- 不允许用于模型选择；
- 不允许回调 GMM 参数。

## 7.4 推荐额外指标

### Standardized separation

```text
d_prime =
(mu_AF - mu_nonAF) /
sqrt((var_AF + var_nonAF) / 2)
```

### Boundary drift

```text
delta_t =
t_target_oracle - t_source*
```

### Midpoint drift

```text
delta_mid =
((mu_AF_target + mu_nonAF_target)/2)
-
((mu_AF_source + mu_nonAF_source)/2)
```

### Scale change

```text
target class gap / source class gap
```

## 7.5 必做可视化

每个 source reference space 生成一张图：

```text
x-axis = disease-direction score
density/histogram:
    source non-AF
    source AF
    target non-AF
    target AF

vertical lines:
    source threshold
    target oracle threshold
```

最终论文非常可能需要这张机制图。

## 7.6 Stage 5D 成功判据

如果出现：

```text
direction cosine 很高
但
target class means / variance / midpoint / threshold 明显变化
```

则核心故事成立：

> Stable disease direction does not imply stable decision boundary.

这将直接为 Target Boundary Reconstruction 提供理论动机。

---

# 8. Stage 5D 补充统计稳定性

以下内容不必阻塞主流程，但最终正式论文应补。

## 8.1 Patient-level bootstrap of direction cosine

对患者有放回抽样，建议：

```text
1000 repetitions
```

重新计算：

```text
disease direction
cross-dataset cosine
```

输出95% CI。

禁止 window-level bootstrap 代替患者级 bootstrap。

## 8.2 Label-permutation null

在 post-hoc analysis 中随机打乱 AF/non-AF label：

```text
N permutations
```

观察随机方向之间的 cosine 分布。

目标：

证明：

```text
真实 cosine ≈ 0.94–0.99
```

明显高于随机标签下的方向一致性。

该实验用于机制稳健性，不用于模型选择。

---

# 9. Stage 6 的重新定义

Stage 5B 后，Stage 6 不应机械地“必须先SupCon”。

先通过 Stage 5C + Stage 5D 判断主要瓶颈。

---

# 10. Stage 6 Decision Gate

## Case A：Ranking仍有明显改进空间

例如：

- cross-domain AUROC/AUPRC仍明显下降；
- target class separation较差；
- prototype direction虽方向一致，但正负类沿轴高度重叠；
- P1/H1 ranking仍不足。

进入：

# Stage 6A — Disease Direction Representation Learning

## Case B：Ranking已经很好，主要失败来自threshold drift

例如：

- AUROC/AUPRC较高；
- direction cosine极高；
- target class separation仍良好；
- 但 source threshold 与 target oracle threshold 明显偏移。

优先进入：

# Stage 6B — Target Boundary Reconstruction v2

Representation learning变为辅助增强实验，而不是主线阻塞项。

---

# 11. Stage 6A — Disease Direction Representation Learning

若 Decision Gate 判定需要改善 representation，按以下顺序执行。

## R0

```text
CE
```

当前冻结 baseline。

## R1

```text
CE + SupCon
```

## R2

```text
CE + Prototype / Center Loss
```

## R3

```text
CE + SupCon + Prototype / Center Loss
```

必须独立可开关。

## 11.1 Stage 6A 主要目标

不要以 source accuracy 提升作为成功标准。

重点比较：

```text
Cross-domain AUROC
Cross-domain AUPRC
target pooled separation
target class gap
direction cosine
patient-bootstrap stability
```

由于 Stage 5B cosine 已接近天花板：

> direction cosine只能作为辅助机制指标，不能作为唯一优化目标。

## 11.2 Stage 6A 成功定义

优先考虑：

```text
mean cross-domain AUROC ↑
mean cross-domain AUPRC ↑
```

同时：

```text
source performance不出现灾难性下降
target separation不下降
direction consistency保持或略升
```

不要仅因：

```text
cosine 0.989 -> 0.994
```

就判断representation变好。

---

# 12. Stage 6B — Target Boundary Reconstruction v2

若 Stage 5D 表明主要问题是 boundary shift，则该阶段成为主方法开发重点。

当前 Stage 5 的自由二成分 GMM 保留为：

```text
GMM-v1
```

下一版发展为：

> Source-Anchored Target Disease-Axis Boundary Reconstruction

## 12.1 Subject-balanced target fitting

避免一个长记录患者产生几千窗口后主导目标分布。

第一版优先：

```text
deterministic subject window cap
```

建议候选：

```text
100
200
500
```

只允许在 development transfers 上选择一次，然后冻结。

## 12.2 Source-anchored initialization

利用 source disease-axis statistics：

```text
mu_source_nonAF
mu_source_AF
std_source_nonAF
std_source_AF
```

初始化 target mixture。

必须保留：

```text
random-init GMM
vs
source-anchored GMM
```

的对照。

## 12.3 更推荐的新方向：受约束的轴向平移/缩放模型

Stage 5B 已证明 disease direction 本身稳定，因此目标域不一定需要自由重学两个Gaussian。

优先尝试一个更符合机制假设的受约束模型：

```text
s_target ≈ a * s_source + b
```

其中：

```text
a > 0
```

目标域无标签估计：

```text
a
b
pi_target
```

必要时再增加少量 variance adjustment。

核心思想：

> 保留 source AF/non-AF 的相对疾病结构，只允许目标域沿稳定疾病轴发生平移、缩放和类别先验变化。

这应作为 GMM-v2 / constrained target-axis adaptation 的重点候选。

## 12.4 Reliability fallback

若无标签目标拟合不可靠：

```text
fallback -> strong source prototype threshold P1
```

或主baseline H1，具体规则在development transfer上预注册后冻结。

可靠性可继续参考：

- ΔBIC
- separation
- posterior entropy
- minimum component weight
- initialization agreement
- subject-bootstrap threshold stability

---

# 13. Development / Final protocol继续冻结

已经查看过target结果的：

```text
LTAFDB -> AFDB
CPSC2021 -> SHDB-AF
```

继续作为：

```text
development transfers
```

允许：

- debugging；
- 初步超参数选择；
- boundary model开发；
- reliability rule设计。

其他 source-target pairs 在方法冻结前不得根据target labels反复调整。

---

# 14. 最终跨域主评价指标

## Ranking metrics

用于评价 Disease Direction / Representation：

```text
AUROC
AUPRC
```

## Operating-point metrics

用于评价 Boundary Reconstruction：

```text
Balanced Accuracy
Macro-F1
MCC
Sensitivity
Specificity
Precision
Accuracy
```

论文主表突出：

```text
AUROC
AUPRC
Balanced Accuracy
Macro-F1
MCC
```

辅助增加：

```text
Sensitivity @ Specificity = 0.90
Specificity @ Sensitivity = 0.90
```

---

# 15. 最终方法不能以“每个指标都必须上涨”为调参目标

严禁根据target label人为追求：

```text
每个 transfer
每个 metric
都比baseline高
```

正确成功标准：

### Representation层

```text
12 transfer平均 AUROC ↑
12 transfer平均 AUPRC ↑
多数 transfer 提升
```

### Boundary层

```text
平均 BACC ↑
平均 Macro-F1 ↑
平均 MCC ↑
```

并进行患者级bootstrap检验。

---

# 16. 最终完整Benchmark

方法完全冻结以后运行：

```text
CPSC2021 -> LTAFDB
CPSC2021 -> AFDB
CPSC2021 -> SHDB-AF

LTAFDB -> CPSC2021
LTAFDB -> AFDB
LTAFDB -> SHDB-AF

AFDB -> CPSC2021
AFDB -> LTAFDB
AFDB -> SHDB-AF

SHDB-AF -> CPSC2021
SHDB-AF -> LTAFDB
SHDB-AF -> AFDB
```

AFDB-source继续标记为：

```text
low-resource source-domain stress test
```

---

# 17. 更新后的 TODO 建议

Codex 应在现有 `TODO.md` 中加入或调整为：

```text
[x] Stage 5A — Prototype Direction vs Linear Head
[x] Stage 5B — Four-Dataset Direction Geometry

[ ] Stage 5C — Strong Source Baseline
    [ ] H0 fixed-0.5 head baseline
    [ ] H1 source-val optimized head threshold
    [ ] P0 fixed source prototype threshold
    [ ] P1 source-val optimized prototype threshold

[ ] Stage 5B+ — Source Head vs Shared Cross-Dataset Disease Axis

[ ] Stage 5D — Disease-Axis Distribution & Boundary Shift
    [ ] class-conditional means
    [ ] class-conditional variances
    [ ] class gap
    [ ] pooled separation
    [ ] prevalence
    [ ] source threshold
    [ ] target oracle threshold (post-hoc only)
    [ ] boundary drift
    [ ] density figures
    [ ] optional patient bootstrap
    [ ] optional label permutation

[ ] Stage 6 Decision Gate

[ ] Stage 6A — Representation Learning
    [ ] CE + SupCon
    [ ] CE + Prototype/Center
    [ ] CE + SupCon + Prototype
    [ ] only if justified by Stage 5D

[ ] Stage 6B — Target Boundary Reconstruction v2
    [ ] subject-balanced target fitting
    [ ] source-anchored GMM
    [ ] constrained shift/scale model
    [ ] reliability fallback

[ ] Frozen full cross-domain benchmark
[ ] Patient-level bootstrap
[ ] Final tables and mechanism figures
```

---

# 18. 推荐的新输出目录

```text
outputs/
├── stage5a_head_vs_direction/
├── stage5b_direction_geometry/
├── stage5c_strong_source_baseline/
├── stage5d_axis_distribution_shift/
├── stage6a_representation_learning/
├── stage6b_target_boundary_v2/
├── final_cross_domain_benchmark/
└── final_statistics/
```

Stage 5B+ 可作为 Stage 5B 原目录的新增子分析，不需要单独复制全部embedding。

---

# 19. Codex 工作纪律

Codex继续实验时必须：

1. 首先读取现有项目文档和本文件；
2. 不覆盖Stage 1–5冻结artifacts；
3. Stage 5B+和5D优先复用既有embedding与selected-window manifest；
4. 所有 target label 使用必须明确标记：
   ```text
   post-hoc mechanism analysis only
   ```
5. Stage 5C threshold selection只能读source validation labels；
6. adaptation代码不能访问target label；
7. target oracle threshold只能用于Stage 5D解释，不允许回流方法；
8. 每阶段开始前先提交：
   - 输入文件；
   - 输出文件；
   - 计划修改代码；
   - label access；
   - leakage safeguard；
9. 完成后：
   - unit test；
   - smoke test；
   - formal run；
   - report；
   - TODO更新；
   - clean Git commit。

---

# 20. Codex下一条任务

读完本文档后，**先执行 Stage 5C，不要直接进入 SupCon。**

建议第一条指令：

> 请完整阅读 `EXPERIMENT_PLAN.md`、`EXPERIMENT_PLAN_SUPPLEMENT.md`、`STAGE5A_REPORT.md`、`STAGE5B_REPORT.md`、`EXPERIMENT_PLAN_UPDATE_AFTER_STAGE5B.md` 和当前 `TODO.md`。  
> 从 Stage 5C — Strong Source Baseline 开始。先不要改代码，请先审计现有 score / metric / threshold 相关模块，列出可复用部分，并给出 H0、H1、P0、P1 四个 baseline 的数学定义、阈值选择流程、target-label 隔离方案、预计新增/修改文件和输出artifact。确认后再实施。

---

# 21. 当前项目最重要的研究命题

后续所有方法与实验都应围绕以下命题服务：

```text
不同数据库：
absolute representation shifts

但：
relative AF-vs-nonAF disease direction remains highly stable

因此：
主要跨域问题可能从“方向迁移”转向
“稳定疾病轴上的目标域分布与边界漂移”

最终方法：
Stable Disease Direction
+
Unlabeled Target Boundary Reconstruction
```

如果 Stage 5D 进一步证实该命题，则后续论文的核心创新应优先集中在：

> **利用稳定疾病方向约束无标签目标域边界重构，而不是重新学习整个目标域特征空间。**
