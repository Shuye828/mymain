# Codex 主提示词：MedTS-TTT 跨数据集房颤判别方向学习

你现在是本项目的科研代码协作者。请先完整理解项目目标，再逐阶段实施；不要一次性生成大量未经验证的代码。每完成一个阶段，都要运行最小测试、记录结果、更新文档，并等待下一步指令。

## 1. 项目背景

基础代码来自：

- GitHub：`https://github.com/mingzhi-c/MedTS-TTT`
- 建议固定上游提交：`151b34083cc3030d7b2f7e09f49dda0fafaa2fd0`

MedTS-TTT 原始目标是医疗时间序列在同一数据集内的跨受试者分类。模型包含 Spatiotemporal Tokenizer、Gated Convolutional Backbone（GCB）、Closed-Loop Self-Alignment Test-Time Training（CLSA-TTT）和分类头。

本项目不以质量监控或患者级聚合为核心创新，而是研究：

> 在源数据集 A 上学习“非房颤到房颤”的疾病判别方向，在无标签目标数据集 B 上沿该方向投影样本，并根据 B 自身分布重新估计分类边界，从而实现跨数据集房颤识别。

## 2. 核心任务定义

当前主任务为窗口级二分类：

- 正类：当前 10 秒 ECG 窗口全部处于 AFIB 节律；
- 负类：当前 10 秒 ECG 窗口全部处于非 AF 节律；
- 排除：跨越节律切换点的混合窗口、无可靠注释窗口、显式 AFL/AT/PAT/NOD/J 等容易与 AF 混淆的节律窗口。先做严格版本，后续再扩展为 AF vs all non-AF。

注意：

- 任务是“当前窗口是否为房颤节律”，不是“患者是否有房颤病史”；
- 目标域标签只能用于最终评价和机制分析，不能用于 GMM、阈值选择、模型更新或超参数选择；
- 所有划分必须以患者或记录主体为组，禁止同一患者泄漏到训练、验证和测试集合。

## 3. 数据集与统一输入

计划使用以下四个 PhysioNet 数据集：

1. `ltafdb`：Long-Term AF Database；
2. `cpsc2021`：CPSC 2021；
3. `afdb`：MIT-BIH Atrial Fibrillation Database；
4. `shdb-af`：SHDB-AF v1.0.1。

共同特点：均可得到两通道动态 ECG 和节律注释，但采样率、导联类型、设备、人群和标签体系不同。

统一输入方案：

- 通道数：2；
- 目标采样率：200 Hz；
- 窗口长度：10 秒；
- 每个样本形状：`[2, 2000]`；
- 初始使用不重叠窗口；
- 预处理：去除 NaN/Inf，统一带通滤波 0.5–40 Hz，重采样到 200 Hz；
- 不在预处理阶段重复 z-score，优先保留 MedTS-TTT 模型内部逐通道 z-score；
- 保留每个数据集原始通道顺序，并在元数据中记录通道名；
- 后续做单通道与双通道消融。

重要实现约束：

- 原始 `MedTS_TTT.py` 的位置编码最多支持 256 个 token；
- 200 Hz × 10 秒 = 2000 点，patch size=8 时得到 250 个 token，能够直接使用；
- 首版配置：`dim=128, num_heads=8, num_layers=6, patch_size=8, num_classes=2, max_channel=2`。

## 4. 数据工程原则

不要把所有 10 秒窗口提前保存成大量独立 `.npy` 文件。优先建立窗口索引表，例如 `data/index/{dataset}_windows.csv`。

每行至少包含：

- dataset
- record_id
- subject_id
- source_path
- fs_original
- channel_names
- start_sample
- end_sample
- rhythm_label
- binary_label
- is_transition
- split
- annotation_source

DataLoader 根据索引按需调用 WFDB 读取片段、滤波、重采样并返回张量。允许后续增加 Zarr、LMDB 或 memmap 缓存，但必须保证可重建、可追踪和不泄漏。

为避免单个 24 小时记录产生过多窗口并主导训练：

- 配置 `max_windows_per_subject_per_class`；
- 使用患者级或记录级平衡采样；
- 所有随机抽样固定 seed；
- 报告每个数据集、患者和类别的窗口数量。

## 5. 模型设计

### 5.1 Backbone

尽量保留上游实现，进行向后兼容重构：

- 增加 `forward_features(x)`，返回 token 平均池化后的 backbone embedding；
- 原 `forward(x)` 仍返回分类 logits；
- 可选 `forward(x, return_features=True)` 返回 logits 与特征；
- 不改变 CLSA-TTT 的数学过程；
- 为“去除 CLSA-TTT”消融提供明确开关。

### 5.2 特征投影与源域训练

在 backbone embedding 后增加：

- 线性或两层 MLP projection head；
- 投影维度初始设为 64；
- 输出做 L2 normalization。

源域训练损失：

`L = L_CE + λ_supcon * L_SupCon + λ_proto * L_Proto`

第一阶段先实现 CE baseline，再逐步加入 SupCon 和 Prototype/Center loss，不能一次混合后无法定位问题。

源域类别原型：

- `c_nonaf = mean(z | y=0)`
- `c_af = mean(z | y=1)`

疾病判别方向：

- `d = normalize(c_af - c_nonaf)`

原型和方向只能由源域训练集特征估计。源域验证集仅用于早停和超参数选择。

### 5.3 目标域边界重构

对目标域无标签样本：

1. 使用 MedTS-TTT 提取目标特征 `z_B`；
2. 计算一维判别得分 `s_B = z_B · d`；
3. 在所有无标签目标得分上拟合双成分一维 GMM；
4. 均值较大的分量映射为 AF，较小分量映射为 non-AF；
5. 使用 GMM 后验概率或两分量密度交点完成预测；
6. 不得读取目标真实标签参与上述步骤。

GMM 初始配置：

- `n_components=2`
- full covariance（在一维等价于分别估计方差）
- `n_init=20`
- `reg_covar=1e-4`
- 固定 random_state

同时拟合单成分 GMM，记录：

- BIC(1-component)
- BIC(2-component)
- `ΔBIC = BIC_1 - BIC_2`
- 两分量均值间距
- pooled separation / effect size
- 后验熵
- 不同初始化稳定性

若双成分结构不可靠，先只记录失败，不要利用目标标签修正。后续再设计回退策略。

## 6. 实验协议

### 6.1 开发顺序

阶段 0：环境和仓库审计  
阶段 1：四个数据集的 WFDB 读取与注释解析  
阶段 2：统一窗口索引和 DataLoader  
阶段 3：复现 Source-only MedTS-TTT 二分类 baseline  
阶段 4：导出 embedding、源原型和疾病方向  
阶段 5：实现目标域 GMM 边界重构  
阶段 6：加入 SupCon/Prototype loss  
阶段 7：完整消融、跨数据集实验和统计分析

### 6.2 推荐首轮迁移组合

先做两个最小闭环：

1. `CPSC2021 -> SHDB-AF`
2. `LTAFDB -> AFDB`

然后扩展到四个数据集之间的全部有向迁移组合。注意 AFDB 只有少量有信号记录，不优先作为深模型从头训练的主源域。

### 6.3 基线和消融

至少实现：

- B0：Source-only GCB，无 CLSA-TTT；
- B1：Source-only MedTS-TTT；
- B2：MedTS-TTT + 源域固定分类头；
- B3：MedTS-TTT + 源疾病方向 + 源固定阈值；
- B4：MedTS-TTT + 源疾病方向 + 目标 GMM 边界；
- B5：GCB + 疾病方向 + 目标 GMM；
- B6：MedTS-TTT + SupCon/Prototype + 疾病方向 + 目标 GMM。

后续再加入 T3A、TENT、SHOT 等强基线。

### 6.4 评价指标

窗口级：

- AUROC
- AUPRC
- Accuracy
- Balanced Accuracy
- Macro-F1
- Sensitivity / Recall
- Specificity
- Precision
- MCC
- 混淆矩阵

机制分析：

- 源域与目标域真实疾病方向余弦相似度（目标标签只在实验后计算）；
- 源/目标类中心漂移；
- 源固定阈值与目标 GMM 阈值差异；
- GMM ΔBIC、分离度和后验熵；
- TTT 前后 embedding 分布及方向稳定性。

### 6.5 两种目标域协议

实现并明确区分：

1. Transductive：同一批无标签目标样本用于拟合 GMM 和最终评价；
2. Inductive holdout：目标患者划分为无标签 adaptation 子集和独立 evaluation 子集，只在 adaptation 子集拟合 GMM。

主要结果必须写清楚使用哪一种。

## 7. 工程结构建议

```text
.
├── MedTS_TTT.py
├── configs/
│   ├── datasets/
│   ├── experiments/
│   └── model/
├── data/
│   ├── raw/
│   ├── index/
│   └── cache/
├── src/
│   ├── data/
│   │   ├── wfdb_io.py
│   │   ├── rhythm_intervals.py
│   │   ├── build_window_index.py
│   │   ├── preprocessing.py
│   │   └── ecg_dataset.py
│   ├── models/
│   │   ├── medts_ttt_wrapper.py
│   │   ├── projection_head.py
│   │   └── losses.py
│   ├── training/
│   │   ├── train_source.py
│   │   └── checkpointing.py
│   ├── adaptation/
│   │   ├── disease_direction.py
│   │   └── target_gmm.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── mechanism_analysis.py
│   └── utils/
├── scripts/
├── tests/
├── outputs/
├── PROJECT_CONTEXT.md
├── DATASET_CONTRACT.md
├── EXPERIMENT_PLAN.md
└── TODO.md
```

## 8. 编码要求

- Python 类型注解；
- dataclass 或 YAML 管理配置；
- 固定随机种子；
- 日志必须记录代码提交、配置、数据版本、数据统计和运行时间；
- 每个数据集适配器有单元测试；
- 所有划分检查患者泄漏；
- 所有输出保留 subject_id、record_id、window_start；
- 不把绝对数据路径写死；
- 不提交原始数据、缓存、模型权重和大文件到 Git；
- 更新 `.gitignore`；
- 关键函数写 docstring；
- 对异常注释、短记录、缺失通道、NaN/Inf、重采样失败给出明确错误或跳过日志；
- 不允许为了让实验运行而静默修改标签规则。

## 9. 你现在首先要做的事情

先不要实现完整模型。请按以下顺序行动：

1. 检查当前仓库文件和环境；
2. 阅读 `README.md`、`MedTS_TTT.py`、`benchmark/README.md`；
3. 输出你对原始代码输入输出、TTT 更新方式、位置编码限制和需要改造点的总结；
4. 创建 `PROJECT_CONTEXT.md`、`DATASET_CONTRACT.md`、`EXPERIMENT_PLAN.md` 和 `TODO.md`；
5. 创建基础目录和 `.gitignore`；
6. 编写一个只做数据审计的脚本：扫描四个数据集目录，使用 WFDB 读取 header，输出记录数、采样率、通道数、通道名、时长和注释文件是否存在，暂时不要切窗；
7. 为审计脚本写最小测试和运行说明；
8. 展示计划修改的文件列表，再开始编码。

每完成一步，说明：

- 新增或修改了哪些文件；
- 为什么这样设计；
- 如何运行；
- 当前验证结果；
- 尚未解决的问题。
