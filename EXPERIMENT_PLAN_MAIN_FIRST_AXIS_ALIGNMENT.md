# EXPERIMENT_PLAN_MAIN_FIRST_AXIS_ALIGNMENT.md
## AFDB 单源多目标主实验优先计划：疾病轴利用与对齐

> **文档定位**
>
> 本文件是 `EXPERIMENT_PLAN_REVISION_AFDB_SOURCE.md` 之后的最新执行计划。
> 它基于已经完成的 `REVISION_R1_REPORT.md`、`REVISION_R2_REPORT.md`、`REVISION_R3_REPORT.md` 进行下一阶段实验设计。
>
> **本文件的核心调整是：优先完成主方法的可行性与正式测试，系统 baseline、扩展对照和完整消融后置。**
>
> baseline 只是“执行顺序后置”，不是取消。最终论文仍必须补齐公平 baseline、统计检验和消融。
>
> 若本文件与旧文档在 R3 之后的实验优先级上冲突，以本文件为准。

---

# 1. 当前已经冻结的关键结论

## 1.1 LTAFDB 数据版本

后续正式实验统一使用：

```text
LTAFDB-clean1h-v1
```

固定规则：

```text
window_start_seconds >= 3600
```

旧版 `LTAFDB-original` 仅作为历史和最终敏感性分析使用。

## 1.2 AFDB source protocol

AFDB 已完成：

```text
23 subjects
5-fold subject-exclusive OOF
full-source final model
3 seeds: 42 / 2024 / 3407
```

主模型：

```text
M_AFDB seed 42
```

AFDB OOF 强 source 结果：

```text
Head:
AUROC = 0.9799
AUPRC = 0.9675
H1 threshold = -0.145196

Prototype:
AUROC = 0.9777
AUPRC = 0.9538
P1 threshold = -0.044348
```

## 1.3 M_AFDB 中的共享疾病几何

四数据集 disease direction 在同一个 `M_AFDB` 特征空间中高度一致：

```text
window-weighted cosine range = 0.9906–0.9964
subject-equal cosine range   = 0.9921–0.9973
```

AFDB → target disease-direction cosine：

```text
AFDB vs CPSC2021        = 0.9964
AFDB vs LTAFDB-clean1h  = 0.9953
AFDB vs SHDB-AF         = 0.9909
```

## 1.4 Linear head 偏离 shared disease axis

在 `M_AFDB` 中：

```text
mean prototype–prototype cosine = 0.9934
mean head-to-prototype cosine    = 0.8821
head-to-shared-axis cosine       = 0.8843
head-to-shared-axis angle        = 27.84°
```

因此当前核心机制是：

> 跨数据集 AF−non-AF prototype directions 构成高度一致的 shared disease geometry，而 source-trained linear head 相对于该共享疾病轴发生明显旋转。

## 1.5 Boundary adaptation 暂不作为优先主线

R3-D 显示 target oracle 相对 P1 的 BACC headroom 很小：

```text
AFDB -> CPSC2021:       +0.0042
AFDB -> LTAFDB-clean1h: +0.0015
AFDB -> SHDB-AF:        +0.0008
```

所以：

```text
Stage 6B / GMM-v2 / target boundary reconstruction
```

暂时后置。

---

# 2. 新的主实验问题

后续优先回答：

> 既然 source disease axis 与三个 target 的真实 AF−non-AF direction 几乎一致，而 source linear head 明显偏离该 axis，那么主动利用或约束 source disease axis，能否提高 AFDB → 多目标数据库的跨域分类能力？

也就是从：

```text
“发现 shared disease axis”
```

推进到：

```text
“把 shared disease-axis mechanism 转换成真实性能提升”
```

---

# 3. 新执行顺序

```text
Main Phase M1 — Direct Disease-Axis Utilization
        ↓
Main Phase M2 — Learned Disease-Axis Alignment
        ↓
Main Phase M3 — Formal AFDB -> 3 Targets Test
        ↓
Main Phase M4 — Multi-seed Stability
        ↓
Baseline Phase B — Formal baselines / comparisons
        ↓
Ablation Phase A — SupCon / Center / combinations
        ↓
Final Statistics
```

注意：

> **主实验先跑，baseline 后补。**

但主实验训练和参数选择仍然只能使用 AFDB source 信息，不能因为 target 测试结果而逐 target 调参。

---

# 4. Main Phase M1 — Direct Disease-Axis Utilization

## 4.1 目的

在不重新训练 backbone 的前提下，先做低成本桥接实验：

> 如果把决策方向从 source linear head 逐步旋转到 source disease axis，跨域表现是否改善？

## 4.2 定义

已有：

```text
d_head  = normalize(w_AF - w_nonAF)
d_proto = normalize(c_AF - c_nonAF)
```

构建插值方向：

```text
d_alpha =
normalize(
    (1 - alpha) * d_head
    + alpha * d_proto
)
```

其中：

```text
alpha = 0      -> 完全 linear-head direction
alpha = 1      -> 完全 source disease axis
```

第一版固定候选：

```text
alpha ∈ {0.00, 0.25, 0.50, 0.75, 1.00}
```

样本统一使用：

```text
z = normalize(h)
score_alpha = z dot d_alpha
```

## 4.3 Alpha 选择禁止使用 target label

`alpha` 的正式选择只允许使用：

```text
AFDB source OOF
```

冻结选择规则：

1. 计算每个 alpha 的 AFDB OOF AUROC；
2. 保留 AUROC 与最佳值差不超过 0.005 的候选；
3. 在候选中选择 OOF BACC 最高者；
4. 若仍并列，选择更靠近 disease axis 的较大 alpha；
5. 阈值也只用 AFDB OOF 选择；
6. 所有选择 artifact 必须在 target evaluation 前冻结。

## 4.4 OOF 实现原则

由于 AFDB OOF 来自 5 个不同 fold model，不能简单拿 final-model direction 重新给 OOF 打分。

每个 fold 应使用自己的：

```text
d_head_fold
d_proto_fold
```

构建：

```text
d_alpha_fold
```

然后对该 fold held-out subjects 生成 OOF score。

五折合并后再执行：

```text
alpha selection
source-only threshold selection
```

最终 full-source `M_AFDB` 使用相同 alpha：

```text
d_alpha_final =
normalize(
    (1-alpha)*d_head_final
    + alpha*d_proto_final
)
```

## 4.5 M1 target 测试

alpha 和 threshold 冻结后，一次性应用：

```text
AFDB -> CPSC2021 evaluation
AFDB -> LTAFDB-clean1h evaluation
AFDB -> SHDB-AF evaluation
```

第一轮只使用：

```text
seed 42
```

输出：

### Ranking

```text
AUROC
AUPRC
```

### Operating point

```text
Balanced Accuracy
Macro-F1
MCC
Sensitivity
Specificity
Precision
Accuracy
```

---

# 5. Main Phase M1 判定

M1 成功不要求三个 target 每个指标全部上升。

重点看：

```text
mean target AUROC
mean target AUPRC
mean target BACC
mean target Macro-F1
mean target MCC
```

以及：

```text
3个 target 中改善的数量
```

如果：

- 至少 2/3 target 的 AUROC 或 AUPRC 不下降且有改善；
- 三 target 平均 ranking metric 提升；
- operating-point metric 没有明显整体恶化；

则说明 disease-axis utilization 值得进入 learned alignment。

即使 M1 没有明显提升，也仍允许进行一次 M2，因为简单线性插值不等价于训练期 representation/head alignment。

---

# 6. Main Phase M2 — Learned Disease-Axis Alignment

## 6.1 核心思想

源域训练时仍然优化 CE，但增加 source-only disease-axis alignment loss：

```text
L_total = L_CE + lambda_axis * L_axis
```

其中：

```text
L_axis =
1 - cosine(d_head, stopgrad(d_source_axis))
```

## 6.2 Source disease axis

训练过程中：

```text
c_nonAF = source non-AF prototype
c_AF    = source AF prototype

d_source_axis =
normalize(c_AF - c_nonAF)
```

只能使用：

```text
AFDB source labels
```

严禁使用：

```text
CPSC / LTAF / SHDB target labels
```

构造任何训练期 axis。

## 6.3 第一版 prototype 更新方式

第一版优先采用：

### Epoch-level frozen axis

每个 epoch 开始前：

1. 用当前模型对 AFDB capped training cohort 提取 embedding；
2. 计算 `c_nonAF`、`c_AF`、`d_source_axis`；
3. `detach / stop-gradient`；
4. 当前 epoch 内固定该 axis；
5. 训练过程中仅依据 `CE + alignment` 更新模型；
6. 下一 epoch 再重新估计 axis。

这样比 mini-batch prototype 更稳定，也更容易审计。

---

# 7. M2 超参数选择

不允许根据 target 结果选 lambda。

第一版候选：

```text
lambda_axis ∈ {0.01, 0.05, 0.10, 0.20}
```

只使用 AFDB 5-fold OOF 选择。

冻结选择规则：

1. OOF AUROC 相比 CE baseline 下降不超过 0.005；
2. OOF BACC 相比 CE baseline 下降不超过 0.01；
3. 在满足条件的候选中选择 `head-to-source-axis angle` 最小者；
4. 若并列，选择更小 lambda；
5. 所有规则在 target formal run 前冻结。

输出：

```text
lambda
OOF AUROC
OOF AUPRC
OOF BACC
head-axis cosine
head-axis angle
```

---

# 8. M2 final model

lambda 冻结以后：

```text
all 23 AFDB subjects
```

训练最终 full-source Axis-Alignment model。

第一轮：

```text
seed = 42
```

训练 epoch 规则沿用 R2 的 source-only development protocol，不能由 target 决定。

---

# 9. Main Phase M3 — 正式主实验测试

M2 final model 冻结后，进行：

```text
AFDB -> CPSC2021
AFDB -> LTAFDB-clean1h
AFDB -> SHDB-AF
```

## 9.1 第一轮正式测试优先 main method

当前优先级：

> 主实验先测试，baseline 后补。

因此 M3 第一轮只需要输出：

```text
Ours-Axis
```

并可读取必要的既有 frozen reference：

```text
R3 Head
R3 Prototype
```

用于 sanity check。

这一阶段不要额外实现：

- TENT
- SHOT
- T3A
- GMM-v2
- 大量 domain adaptation baseline

这些全部后置。

---

# 10. 主实验需要重点观察的指标

## Primary ranking

```text
AUROC
AUPRC
```

## Primary operating point

```text
Balanced Accuracy
Macro-F1
MCC
```

## Secondary

```text
Sensitivity
Specificity
Precision
Accuracy
```

## Mechanism

同时记录：

```text
head-to-source-axis cosine
head-to-source-axis angle
target class gap
target d-prime
target overlap
```

后 3 项 target 机制指标只在 score/archive 冻结后用 target label 计算：

```text
post-hoc analysis only
```

---

# 11. 主实验成功标准

不要设置“所有 target 所有指标必须上涨”的标准。

### Strong success

```text
3-target mean AUROC ↑
3-target mean AUPRC ↑
3-target mean BACC / Macro-F1 / MCC 中至少2项 ↑
且至少2/3 target有一致改善
```

### Partial success

```text
ranking明显改善
但 operating point变化较小
```

这仍然可以成立，因为 R3 已证明 boundary headroom 很小。

### Failure

如果：

```text
平均 AUROC/AUPRC下降
且
三个target多数 operating metrics也下降
```

则停止 Axis Alignment 主线，转入其他 representation alternatives。

---

# 12. Main Phase M4 — Multi-seed Stability

只有 seed42 主实验完成后再跑：

```text
seed = 2024
seed = 3407
```

目的：

```text
optimization stability
not target-based model selection
```

三个 seed 使用：

- 同一训练规则；
- 同一 lambda；
- 同一 epoch protocol；
- 同一 target evaluation protocol。

禁止根据 target 结果挑 best seed。

输出：

```text
mean ± std
```

至少针对：

```text
AUROC
AUPRC
BACC
Macro-F1
MCC
```

---

# 13. Baseline Phase B — 后置执行

主实验完成以后，再系统补 baseline。

至少包含：

## B0

```text
MedTS-TTT head, fixed threshold
```

## B1

```text
MedTS-TTT head, AFDB OOF calibrated threshold
```

## B2

```text
Source prototype direction, fixed threshold
```

## B3

```text
Source prototype direction, OOF calibrated threshold
```

## B4

```text
M1 direct axis interpolation
```

## B5

```text
Existing unrestricted GMM-v1
```

根据论文定位再考虑：

```text
TENT
T3A
SHOT
其他 TTA / source-free adaptation baseline
```

这些公开方法 baseline 不应阻塞当前 main test。

---

# 14. Ablation Phase A — 主实验后再做

只有在 Axis Alignment 有效后才扩展：

```text
A0: CE
A1: CE + Axis Alignment
A2: CE + SupCon
A3: CE + Prototype/Center
A4: CE + Axis + SupCon
A5: CE + Axis + Prototype
```

重点比较：

```text
cross-domain AUROC/AUPRC
class gap
d-prime
overlap
head-axis angle
```

不要把 source accuracy 当作主要成功标准。

---

# 15. Target 数据访问纪律

虽然三个 target 已在 R3 中用于 post-hoc mechanism analysis，后续仍要避免目标特异性调参。

要求：

1. `alpha` 只由 AFDB OOF 选择；
2. `lambda_axis` 只由 AFDB OOF 选择；
3. target label 只能在 formal frozen evaluation 后使用；
4. 不允许 target-specific lambda/alpha；
5. 如果 main formal run 后方法需要修改：
   - 新方法必须作为新的预注册 variant；
   - 不能隐藏之前失败结果；
   - 不能把 target oracle 信息反馈回训练。

---

# 16. 推荐新增输出目录

```text
outputs/
├── main_m1_axis_interpolation/
│   ├── oof/
│   └── targets/
├── main_m2_axis_alignment/
│   ├── folds/
│   ├── oof/
│   └── final_model/
├── main_m3_afdb_three_target/
│   ├── cpsc2021/
│   ├── ltaf_clean1h/
│   └── shdb/
├── main_m4_multiseed/
├── baselines_late/
└── ablations_late/
```

---

# 17. 推荐新增代码模块

在不破坏现有结构的前提下优先复用已有模块。

建议新增或复用：

```text
src/models/disease_axis.py
src/training/axis_alignment.py
src/evaluation/axis_scores.py

scripts/run_axis_interpolation.py
scripts/train_axis_alignment.py
scripts/evaluate_main_three_targets.py
```

若已有近似功能，优先复用，不重复造轮子。

---

# 18. 推荐 TODO 更新

```text
[x] Revision R1 — LTAFDB-clean1h
[x] Revision R2 — AFDB Source Protocol
[x] Revision R3 — M_AFDB Mechanism Revalidation

[ ] Main M1 — Direct Disease-Axis Utilization
    [ ] fold-specific d_head / d_proto
    [ ] alpha OOF scoring
    [ ] source-only alpha selection
    [ ] source-only threshold selection
    [ ] seed42 3-target test
    [ ] report

[ ] Main M2 — Learned Disease-Axis Alignment
    [ ] axis loss
    [ ] epoch-level source prototype axis
    [ ] AFDB OOF lambda selection
    [ ] final seed42 model
    [ ] report

[ ] Main M3 — AFDB -> 3 Targets Formal Test
    [ ] CPSC2021
    [ ] LTAFDB-clean1h-v1
    [ ] SHDB-AF
    [ ] unified metrics
    [ ] mechanism metrics
    [ ] report

[ ] Main M4 — Multi-seed Stability
    [ ] seed2024
    [ ] seed3407
    [ ] mean ± std

[ ] Baseline Phase B
[ ] Ablation Phase A
[ ] Patient-level bootstrap
[ ] Final tables
[ ] Final figures
```

---

# 19. Codex 当前第一条任务

**不要先补 baseline。**

首先完成：

```text
Main M1 — Direct Disease-Axis Utilization
```

但第一步只做实施审计，不立即大范围修改代码。

Codex 应先：

1. 阅读全部最新实验文档和 R1–R3 reports；
2. 找到 AFDB fold checkpoints、fold-specific classifier head、final `M_AFDB`、R2 OOF archive、R3 disease-direction implementation；
3. 判断能否复用已有 artifact；
4. 设计 fold-specific `d_alpha` OOF scoring；
5. 冻结 alpha selection rule；
6. 冻结 OOF threshold selection rule；
7. 明确 target evaluation split；
8. 列出新增/修改文件；
9. 明确所有 label access；
10. 等待确认后再实施。

---

# 20. 当前阶段核心研究目标

后续主实验不再围绕“目标阈值重构”展开，而是围绕：

> **AFDB source disease axis 已被证明与三个 target 的真实疾病方向高度一致。下一步需要验证：将 source classifier 的判别方向显式向这个稳定 disease axis 靠拢，是否能够提高跨数据集 AF 分类性能。**

最终希望形成：

```text
Shared disease geometry discovery
        ↓
Source disease-axis utilization
        ↓
Axis-aligned source classifier
        ↓
AFDB -> 3 heterogeneous targets
        ↓
Improved cross-domain robustness
```
