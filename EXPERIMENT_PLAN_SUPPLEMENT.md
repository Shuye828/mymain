# EXPERIMENT_PLAN_SUPPLEMENT.md
## 跨数据集房颤判别方向学习：Stage 5 之后的补充实验计划

> **用途**  
> 本文档是现有项目文档与 `TODO.md` 的**补充文件**，用于指导 Stage 5 之后的实验。  
> **不替代**此前的数据协议、标签规则、数据集划分、Stage 1–5 已冻结结果与可复现约束。  
> 若本文档与旧 `TODO.md` 在 Stage 5 之后的任务优先级上冲突，以本文档为准；Stage 1–5 的既有实现与结果保持冻结，不回改结果。

---

# 1. 当前实验状态

项目已经完成：

- Stage 1：四个数据集读取、节律区间解析与严格标签映射；
- Stage 2：患者级划分、10 s 非重叠窗口、按需预处理与目标标签隐藏；
- Stage 3：CPSC2021 与 LTAFDB 的 CE-only MedTS-TTT source baseline；
- Stage 4：源域 embedding、AF / non-AF prototype 与疾病判别方向导出；
- Stage 5：无标签目标域一维 GMM 边界重构，已完成：
  - `LTAFDB -> AFDB`
  - `CPSC2021 -> SHDB-AF`
  - inductive holdout 与 transductive 两种协议。

Stage 5 的当前结果表明：

1. MedTS-TTT 源模型具有较强的跨数据集 AF 排序能力；
2. 使用 `c_AF - c_nonAF` 得到的一维疾病方向后，跨数据集 AUROC 与原线性分类头非常接近；
3. GMM 能够改变目标域工作点，但尚未在所有指标、所有迁移任务上稳定超过 source baseline；
4. 因此下一阶段的重点不再是继续堆叠复杂模块，而是先验证“疾病方向”的本质、跨域稳定性与可学习性。

---

# 2. 重新确认的论文主线

本项目后续统一围绕以下假设开展：

> 不同 ECG 数据集之间存在明显的绝对分布偏移，但 AF 与 non-AF 之间的**相对疾病判别方向**可能比固定分类边界具有更好的跨数据集稳定性。  
> 因此，本项目希望显式学习跨域稳定的疾病方向，再利用无标签目标域数据重构目标特异性的决策边界。

整体框架：

```text
ECG
  ↓
MedTS-TTT backbone
  ↓
Disease Direction Learning
  ↓
Target Boundary Reconstruction
  ↓
AF / non-AF
```

**MedTS-TTT 在后续实验中固定作为 backbone 和 source-only baseline。**

当前阶段不再新增：

- w/o CLSA-TTT；
- GCB-only；
- Vanilla TTT；
- MedTS-TTT 内部机制消融。

除非后续论文审稿明确要求，否则不把“MedTS-TTT 本身为什么有效”作为本项目核心实验问题。

---

# 3. 后续实验的四个核心目标

## Goal A：判断“疾病方向”和线性分类头是否本质等价

当前疾病方向：

\[
d_{\mathrm{proto}}
=
\frac{c_{\mathrm{AF}}-c_{\mathrm{nonAF}}}
{\|c_{\mathrm{AF}}-c_{\mathrm{nonAF}}\|}
\]

MedTS-TTT 二分类线性头的判别方向：

\[
d_{\mathrm{head}}
=
\frac{w_{\mathrm{AF}}-w_{\mathrm{nonAF}}}
{\|w_{\mathrm{AF}}-w_{\mathrm{nonAF}}\|}
\]

必须首先判断二者是否只是同一判别轴的不同写法。

### 必做指标

对每一个 source checkpoint，至少计算：

1. `cosine(d_proto, d_head)`
2. 两方向夹角（degree）
3. prototype score 与 classifier logit-difference 的 Pearson correlation
4. prototype score 与 classifier logit-difference 的 Spearman correlation
5. 两种 score 在：
   - source validation
   - source test
   - available cross-domain target evaluation
   上的 AUROC / AUPRC

其中：

```text
classifier_score = logit_AF - logit_nonAF
prototype_score  = z @ d_proto
```

### 解释规则

若：

```text
cosine > 0.95
且
Spearman > 0.98
```

则当前 `c_AF - c_nonAF` 不能单独作为核心创新。

此时后续论文创新应明确转向：

> “显式学习跨域稳定的 disease direction”，而不是“计算两个 prototype 的差”。

若两方向相似性不高，但 prototype direction 跨域性能更好，则保留并进一步分析其几何差异。

### 输出

新增：

```text
outputs/analysis/head_vs_direction/
    cpsc2021/
    ltafdb/
```

每个目录至少包含：

- `direction_comparison.json`
- `score_correlation.json`
- `split_metrics.csv`
- 必要的散点图/直方图

---

# 4. Goal B：构建四数据集疾病方向相似性图谱

本项目使用：

- LTAFDB
- CPSC2021
- AFDB
- SHDB-AF

目标不是只做两两迁移，而是判断四个数据库的 AF / non-AF 相对方向是否存在整体稳定结构。

## 4.1 重要约束：方向必须在同一个 feature space 中比较

禁止直接比较：

```text
CPSC 模型空间中的 d_CPSC
vs
LTAF 模型空间中的 d_LTAF
```

因为不同训练模型的 128 维坐标系不可直接比较。

正确方法：

### Reference space 1：CPSC2021 source model

冻结 `M_CPSC`，分别提取：

```text
CPSC2021
LTAFDB
AFDB
SHDB-AF
```

的 embedding。

使用各数据集真实标签**仅用于事后机制分析**，计算：

```text
d_CPSC^M_CPSC
d_LTAF^M_CPSC
d_AFDB^M_CPSC
d_SHDB^M_CPSC
```

再构建 4 × 4 cosine similarity matrix。

### Reference space 2：LTAFDB source model

同样构建：

```text
d_CPSC^M_LTAF
d_LTAF^M_LTAF
d_AFDB^M_LTAF
d_SHDB^M_LTAF
```

得到第二个 4 × 4 similarity matrix。

后续若完成 SHDB-AF source model，可增加第三个 reference space。

AFDB source model 因受试者数少，暂作为低资源补充实验，不是第一优先级。

---

# 5. Goal C：同时证明“域差异大，但疾病方向稳定”

只证明方向相似还不够，需要同时证明四个数据集确实存在明显 domain shift。

## 必做分析

在同一个冻结 reference model 的 feature space 中，对每个数据集计算：

### 1. 全局特征中心

\[
m_D=\frac{1}{N_D}\sum_i z_i
\]

### 2. 数据集间绝对中心距离

\[
\|m_A-m_B\|_2
\]

### 3. 可选域差异指标

优先选择 1–2 个即可：

- MMD；
- CORAL distance；
- centroid distance；
- domain classifier accuracy（可选）。

### 4. 疾病方向相似度

\[
\cos(d_A,d_B)
\]

最终希望形成如下证据：

```text
Absolute domain distribution:
A 与 B 差异明显

Relative disease structure:
AF - nonAF 的方向仍高度相似
```

### 建议最终图

1. `4 × 4 Direction Cosine Similarity Heatmap`
2. `4 × 4 Domain Centroid Distance Heatmap`
3. 两张图并排比较

理想论文叙事：

> 数据集绝对位置发生明显变化，但疾病相对方向保持稳定。

---

# 6. Goal D：建立明确、公平的跨域 baseline

论文主 baseline 为：

> **Source dataset 训练 MedTS-TTT，参数冻结，直接应用于 target dataset。**

必须至少建立两个版本：

## Baseline-0：固定 0.5 阈值

```text
source train
→ MedTS-TTT
→ target inference
→ P(AF) > 0.5
```

## Baseline-1：Source-validation optimized threshold

仅使用 source validation label 选择：

```text
t_source*
```

建议主规则：

```text
argmax Balanced Accuracy
```

同时可保存 source-val Macro-F1 optimal threshold 作为辅助。

之后：

```text
t_source*
```

完全冻结，直接应用 target。

**禁止使用 target label 选阈值。**

论文中 Baseline-1 应作为比 Baseline-0 更强、更公平的主对照。

---

# 7. 后续模型必须解决两个不同层面的问题

完整方法分成两部分：

## Part 1：Disease Direction Learning

目标：

> 提高跨域 ranking performance。

重点指标：

- AUROC
- AUPRC
- direction cosine similarity

当前 Stage 4 的 prototype direction 只是起点。

后续依次实现：

### R0
```text
CE
```

### R1
```text
CE + SupCon
```

### R2
```text
CE + Prototype/Center Loss
```

### R3
```text
CE + SupCon + Prototype/Center Loss
```

每个 loss 必须独立可开关。

不要一次性把多个 loss 混合后再判断结果。

### Representation experiment 的判断标准

Source performance 不要求进一步提高，只要求不能出现灾难性下降。

真正重点观察：

1. Cross-domain AUROC 是否提升；
2. Cross-domain AUPRC 是否提升；
3. 4-domain direction cosine similarity 是否提升；
4. target disease score separation 是否提升；
5. target boundary reconstruction 后 BACC/F1/MCC 是否进一步提升。

---

# 8. Target Boundary Reconstruction 的定位

Target GMM 不负责改善排序能力。

若 B3 与 B4 使用同一个 disease score：

```text
score = z @ d
```

则 AUROC / AUPRC 应使用同一个 score 计算。

GMM 只负责产生：

```text
target-specific threshold
```

因此后续正式报告必须区分：

## Ranking metrics

- AUROC
- AUPRC

用于评价 Disease Direction Learning。

## Threshold / operating-point metrics

- Balanced Accuracy
- Macro-F1
- MCC
- Sensitivity
- Specificity
- Precision
- Accuracy

用于评价 Target Boundary Reconstruction。

**不要再使用 GMM posterior 作为主 AUROC/AUPRC score 来解释疾病排序变化。**

---

# 9. Target GMM 下一版改进

Stage 5 GMM 保留为初始版本，但下一版重点考虑：

## 9.1 Subject-balanced fitting

当前一个患者可能提供数千个高度相关窗口。

后续 GMM 拟合不能让长记录患者主导总体分布。

优先实现以下任一方式：

### 方案 A：subject window cap

每个 adaptation subject 最多随机保留：

```text
K = 100 / 200 / 500
```

个窗口参与 GMM。

### 方案 B：subject-equal weighting

若 sklearn GMM 不方便 sample_weight，则可先使用确定性 subject-balanced subsampling。

第一版优先采用方案 A，便于审计和复现。

## 9.2 Source-anchored initialization

当前 GMM 可继续使用多次初始化，但增加一个 source-aware initialization 分支。

目标不是固定 source threshold，而是给两个 component 提供疾病语义锚点。

例如：

```text
source non-AF projection mean
source AF projection mean
```

经过目标域整体中心平移后作为初始 component means。

必须保留：

```text
random-init GMM
vs
source-anchored GMM
```

对照。

## 9.3 Reliability fallback

若 GMM 不满足预先冻结的可靠性标准，则：

```text
fallback → source-validation threshold
```

可靠性条件可以沿用 Stage 5：

- ΔBIC
- separation
- posterior entropy
- minimum component weight
- initialization agreement

但阈值必须只在 development transfers 上确定，然后冻结。

---

# 10. 四数据集完整跨域 benchmark

最终需要评估 4 个数据集之间所有有向迁移：

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

共 12 个 source-target transfer。

### 注意

AFDB source 只有少量有效受试者，最终应标记为：

> low-resource source-domain stress test

不要与另外三个源域完全等价解读。

---

# 11. Development transfers 与 final frozen transfers

为避免人工使用 target label 反复调参造成隐式 target overfitting：

## 已经看过标签结果的 development transfers

固定为：

```text
LTAFDB -> AFDB
CPSC2021 -> SHDB-AF
```

允许用于：

- debug；
- SupCon / Prototype loss 超参数初筛；
- GMM 改进；
- reliability rule 设计。

## Final frozen transfers

其余 transfer 在方法与超参数冻结前：

- 不根据 target label 反复调参；
- 不使用 target label 选择模型；
- 不使用 target label 调整阈值；
- 不使用 target label 决定 GMM component mapping。

方法冻结后一次性运行完整 benchmark。

---

# 12. 核心评价指标

最终主表只突出以下 5 个：

```text
AUROC
AUPRC
Balanced Accuracy
Macro-F1
MCC
```

辅助报告：

```text
Accuracy
Sensitivity
Specificity
Precision
Confusion Matrix
```

## 医学工作点补充指标

建议增加：

```text
Sensitivity @ Specificity = 0.90
Specificity @ Sensitivity = 0.90
```

用于减少单一阈值导致的 Sensitivity / Specificity trade-off 争议。

---

# 13. 最终“成功”的定义

不能通过实验设计保证所有指标一定升高，也不允许通过 target label 调参去“保证”结果。

预先定义成功标准：

## Disease Direction Learning 成功

相较 Source-only MedTS-TTT baseline：

- 平均 cross-domain AUROC 提升；
- 平均 cross-domain AUPRC 提升；
- 多数 transfer 上方向一致性提高；
- 结果在多个 source-target pair 上稳定，而不是只在单一任务提升。

## Target Boundary Reconstruction 成功

相较 frozen source threshold：

- 平均 BACC 提升；
- 平均 Macro-F1 提升；
- 平均 MCC 提升；
- Sensitivity / Specificity 的变化符合新的 operating point，而不是只追求单边指标。

## 最终完整方法成功

希望：

```text
Ours > Baseline
```

在 12 个 transfer 的平均：

- AUROC
- AUPRC
- BACC
- Macro-F1
- MCC

上均取得稳定改善。

强调“平均与多数任务稳定提升”，不要人为追求每个单独 transfer 的所有指标全部提高。

---

# 14. 统计分析

正式结果必须采用**患者级 bootstrap**，禁止 window-level bootstrap。

建议：

```text
1000 bootstrap repetitions
```

每次：

1. 从 target evaluation subjects 中有放回采样患者；
2. 保留被采中患者的全部 evaluation windows；
3. 分别计算 baseline 与 ours；
4. 记录：

```text
ΔAUROC
ΔAUPRC
ΔBACC
ΔMacro-F1
ΔMCC
```

最终报告：

- mean delta
- 95% confidence interval
- 置信区间是否跨 0

---

# 15. 下一阶段的严格执行顺序

后续请按以下顺序更新 `TODO.md`，不要跳步。

## Stage 5A — Direction vs Linear Head

### 任务
- 提取二分类头方向；
- 与 prototype direction 比较；
- 计算 cosine / angle；
- 计算 Pearson / Spearman；
- 比较 source 与 target ranking metrics。

### 完成标准
- CPSC source checkpoint 完成；
- LTAF source checkpoint 完成；
- 输出统一表格；
- 明确判断“高度等价 / 部分等价 / 明显不同”。

## Stage 5B — Four-Dataset Direction Geometry

### 任务
- 在 `M_CPSC` feature space 中提取四数据集真实 disease direction；
- 在 `M_LTAF` feature space 中重复；
- 构建 4 × 4 direction cosine matrix；
- 构建 4 × 4 domain centroid distance matrix；
- 可选计算 MMD。

### 完成标准
- 所有方向都在同一冻结模型空间内计算；
- target label 仅用于 post-hoc mechanism analysis；
- 输出 heatmap 与 CSV；
- 不修改模型。

## Stage 5C — Strong Source Baseline

### 任务
- 保留 0.5 baseline；
- 在 source validation 上冻结 `t_source*`；
- 把 `t_source*` 应用全部 target；
- 统一 B0/B1 命名和输出。

### 完成标准
- threshold 选择代码禁止读取 target labels；
- result manifest 写明 threshold 来源。

## Stage 6A — CE + SupCon

### 任务
- 实现 SupCon；
- 独立开关；
- tiny-overfit；
- source formal run；
- direction export；
- development transfer evaluation。

### 判断
优先看：

```text
cross-domain AUROC/AUPRC
direction similarity
```

而不是 source accuracy。

## Stage 6B — CE + Prototype/Center

同上，必须与 SupCon 分开完成。

## Stage 6C — CE + SupCon + Prototype

只有在 6A / 6B 至少一个方向有效时再执行。

## Stage 7 — Target Boundary Reconstruction v2

实现：

- subject-balanced target fitting；
- source-anchored GMM；
- reliability fallback；
- primary inductive protocol；
- secondary transductive protocol。

## Stage 8 — Frozen 4-Dataset Benchmark

在方法和超参数完全冻结后执行剩余 source-target transfers。

## Stage 9 — Statistical Analysis and Final Tables

- patient-level bootstrap；
- 主结果表；
- direction geometry 表；
- ablation 表；
- GMM operating-point 表；
- final figures。

---

# 16. 推荐新增文件

在不破坏现有项目结构的前提下，建议新增：

```text
src/analysis/
    head_direction_equivalence.py
    cross_dataset_direction_geometry.py
    domain_shift.py

src/evaluation/
    threshold_selection.py
    operating_points.py
    bootstrap.py

scripts/
    analyze_head_vs_direction.py
    analyze_four_dataset_directions.py
    build_source_thresholds.py
    run_cross_domain_benchmark.py

configs/analysis/
    direction_geometry.json
    domain_shift.json
```

若现有项目已经有相近模块，应复用，不重复造轮子。

---

# 17. 结果文件命名建议

统一使用：

```text
outputs/
    stage5a_head_vs_direction/
    stage5b_direction_geometry/
    stage5c_source_threshold_baseline/
    stage6a_supcon/
    stage6b_prototype/
    stage6c_supcon_prototype/
    stage7_target_boundary_v2/
    stage8_cross_domain_benchmark/
    stage9_statistics/
```

每次正式实验必须保存：

- config
- Git commit
- dataset/index hash
- source checkpoint hash
- target split version
- metrics
- predictions/scores
- subject/record/window identity
- runtime
- seed

---

# 18. Codex 执行原则

接下来 Codex 必须遵循：

1. **先阅读现有项目文档、`TODO.md`、Stage 1–5 report，再读本补充文档。**
2. 不删除、不覆盖 Stage 1–5 frozen artifacts。
3. 新阶段优先新增独立分析脚本，不修改已有正式结果。
4. 每个阶段开始前，先汇报：
   - 计划修改的文件；
   - 输入 artifact；
   - 输出 artifact；
   - 是否使用 target label；
   - 如何保证不泄漏。
5. 每个阶段结束后，必须：
   - 跑测试；
   - 跑最小真实数据 smoke test；
   - 输出 report；
   - 更新 `TODO.md`；
   - 提交干净 Git commit。
6. 任何 target label 使用都必须明确标记：
   - `adaptation-time prohibited`
   - 或 `post-hoc analysis only`
7. 不允许为了让结果更好而：
   - 根据 target label 选择阈值；
   - 根据 target label 调 GMM；
   - 根据 target label 选择 direction；
   - 根据 final target transfer 反复调 loss 权重。

---

# 19. Codex 下一条具体任务

读完本文档后，**不要直接开始 Stage 6 训练**。

先完成：

```text
Stage 5A — Direction vs Linear Head
Stage 5B — Four-Dataset Direction Geometry
Stage 5C — Strong Source Baseline
```

其中最优先是 Stage 5A。

第一条实施任务：

> 基于已经冻结的 CPSC2021 和 LTAFDB Stage 3 best checkpoints，以及 Stage 4 disease-direction artifacts，新增一个只读分析流程，比较 prototype disease direction 与 binary linear classifier direction。先列出计划修改文件、数学定义、输入输出和泄漏检查方案，不要立即大范围修改代码。

---

# 20. 当前阶段最终目标

接下来不是立即追求复杂模型，而是先回答三个问题：

### Q1
当前 prototype disease direction 是否只是线性分类头的等价表示？

### Q2
在同一冻结 feature space 中，四个数据集的 AF / non-AF disease direction 是否保持较高一致性？

### Q3
不同数据集的绝对 feature distribution 是否明显不同，而 disease direction 仍相对稳定？

只有这三个问题得到清晰答案后，再决定 Stage 6 的 Disease Direction Learning 应该重点优化什么。
