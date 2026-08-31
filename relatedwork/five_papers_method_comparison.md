# 一、统一方法对比表

| 论文 | 核心问题 | 核心思想 | 主干网络 | 缺失处理 | 跨模态交互 | 图/超图结构 | 是否补全 | 是否蒸馏 | 训练协议 | 单模型支持全部组合 | 主要局限 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmFormer (MICCAI 2022) | 输入子集变化使融合关系失真并加剧强模态依赖 | 模态特异 CNN/Transformer 编码后，以 mask-aware inter-modal Transformer 融合 | 4 个 3D CNN encoder + intra/inter-modal Transformer + CNN decoder | 缺失分支 feature 清零并从 attention 中屏蔽；不生成 | bottleneck token attention，多尺度 skip 拼接 | 无显式图；attention 是隐式成对关系 | 否 | 否 | 完整四模态病例，每样本随机非空 Bernoulli mask，联合辅助分割监督 | 是，覆盖 15 个非空组合 | 训练依赖完整数据；参数/计算大；无显式 anatomy 结构 |
| M3AE (AAAI 2023) | 随机缺失导致表示不完整，普通补零不能恢复共享语义 | 高比例 modality+patch masked autoencoding 预训练，model inversion 学全局替代 token，fine-tuning 做自蒸馏 | 3D U-Net/autoencoder | 用可学习 dataset-level `x_sub` 替代缺失/遮蔽位置 | 主要通过联合重建形成共享潜空间；非显式 pairwise graph | 无图/超图 | 是，属于替代表示/重建，不是患者特异 MRI 生成 | 是，同一患者两种缺失视图的 bottleneck self-distillation | 完整多模态预训练；87.5% 组合遮蔽；微调随机两种缺失组合 | 是，评估 15 种组合 | `x_sub` 是全局模板而非患者特异；两阶段训练；图结构缺失 |
| ReHyDIL (MICCAI 2025) | 模态顺序到达时的新域学习、旧域遗忘和小病灶不平衡 | replay + 跨患者动态超图 + Tversky-aware contrastive learning | 2D U-Net/CHSNet | 不补全；用旧样本 replay 和跨患者传播保持知识 | batch 中患者-空间 feature 经超图传播；TAC 对齐旧新模态 | 真正超图；动态 KNN 超边，跨患者与空间顶点混合 | 否 | 非 teacher-student；有 prior-model prediction queue/对比迁移 | T1→T2→FLAIR→T1ce domain incremental；每阶段保留约 10% replay | 处理连续模态域，但不等同普通任意组合统一 mask 模型 | batch/test 依赖；2D；动态图受缺失扰动；DIL 与目标任务设定不同 |
| D3Seg (arXiv v2, 2026，预印本) | 可用模态依赖变化、关键 latent 缺失及错误补全 | 多跳 modality graph 重估依赖，条件扩散补 T1ce/FLAIR latent，两级 refinement 纠错 | 独立 3D encoders + Transformer/Mamba + decoder | mask graph；对缺失 T1ce/FLAIR 做 latent diffusion；其余不补 | MMGF 多跳图、bottleneck cross-attention、Mamba skips | 普通四节点模态图；单患者、动态成对邻接 | 是，选择性 latent 补全 | 否 | 报告 15 组合；mask 采样、总损失权重和单 checkpoint 细节需核查 | 文意指向是，但协议需进一步核查 | 未正式发表/无官方代码；扩散推理昂贵；EGDR 分割特化 |
| AdaMM (Medical Image Analysis 2026) | 完整与缺失输入之间的知识、分布和关系落差 | full-modal teacher 向 missing-modal student 多层蒸馏，组合 adapter + BBDM + GARM + LGRM | 独立 teacher/student encoder-decoder | 缺失通道置零；按组合选择 adapter；不生成 | GSME 跨特征关系、teacher-student alignment、anchor graph 交互 | 普通 anchor graph；单患者内 `CF/GF` 两图以 GCN/GAT传播 | 否 | 是，feature/probability/adversarial distillation | 每迭代随机 15 组合之一；teacher 用完整输入，student 用缺失输入 | 是，但含 14 个组合专属 adapters | 依赖完整数据和 teacher；组合数指数增长；当前官方代码快照不完整 |

**统一判断。** 五篇方法没有一篇同时实现“患者内 anatomy/ROI 超图 + 任意缺失模态 + 患者级分类”。ReHyDIL 是唯一真正使用超图者，但其节点、超边和训练协议都与目标方案不同；D3Seg、AdaMM 使用普通图；mmFormer 的 attention 和 M3AE 的共享潜空间都不能写成超图。

# 二、五篇方法的本质差异

## 2.1 它们怎样定义“缺失模态问题”

### Transformer 跨模态相关性建模：mmFormer

mmFormer 把问题定义为**输入集合改变后，融合关系仍应稳定**。它不追问缺失图像长什么样，而是让每个模态先独立形成可用表示，再让可用 token 在 inter-modal Transformer 中重建共享语义。缺失鲁棒性来自“减少对单一强模态的依赖”和“让注意力只在观察证据之间重组”。

### Masked autoencoder 表征预训练：M3AE

M3AE 把问题定义为**模型没有学会从严重不完整上下文中恢复联合表示**。高比例 modality/patch mask 迫使 encoder 学习冗余与互补关系；model inversion 学到的 `x_sub` 提供缺失位置的通用替代先验；self-distillation 进一步要求同一患者不同缺失视图一致。它更接近“通过预训练形成可恢复的共享空间”。

### Hypergraph + domain incremental learning：ReHyDIL

ReHyDIL 把问题定义为**模态域按顺序到达时持续学习且不能遗忘**。缺失不是一次随机 mask，而是某阶段只能看到当前/回放模态。跨患者超图用于从 batch 中提取高阶共性，replay 和 TAC 用于保存旧域知识及处理前景不平衡。其核心变量是时间顺序和记忆，而不是任意组合内的患者级融合。

### Graph dependency + diffusion imputation：D3Seg

D3Seg 把问题拆成**观察模态依赖变化**与**关键语义真的不存在**。前者由动态 modality graph 解决，后者由 T1ce/FLAIR latent diffusion 解决；re-masking 和 EGDR 对生成与分割错误再纠正。它是五篇中最明确的“关系建模 + 显式潜补全”双路径方法。

### Graph refinement + knowledge distillation：AdaMM

AdaMM 把问题定义为**full-modal expert 与 missing-modal learner 之间的知识鸿沟**。它不重建缺失模态，而是让 student 在输出、bottleneck 分布和 feature graph 上接近 teacher；组合 adapter 负责不同缺失模式的域偏移。其关键资源是训练期完整模态配对与 teacher。

## 2.2 范式差异的关键轴

| 轴 | mmFormer | M3AE | ReHyDIL | D3Seg | AdaMM |
|---|---|---|---|---|---|
| 主要依靠什么弥补缺失 | 观察模态重组 | 预训练共享表示与全局替代 token | 旧样本记忆与跨患者高阶传播 | 观察关系重组 + 关键 latent 生成 | 完整 teacher 的知识迁移 |
| 关系是显式还是隐式 | attention，隐式 | 潜空间，隐式 | 超图，显式高阶 | 普通模态图，显式成对 | 普通 anchor graph，显式成对 |
| 关系范围 | 单患者、跨模态/空间 token | 单患者重建上下文 | 跨患者且含空间顶点 | 单患者、模态节点 | 单患者、feature prototype |
| 是否要求患者特异补全 | 否 | 否，`x_sub` 是全局替代 | 否 | 是，条件生成 latent | 否 |
| 最主要训练依赖 | 完整四模态随机 mask | 完整数据的预训练/重建 | 顺序模态阶段与 replay | 完整目标 latent | 完整 teacher 输入 |
| 对分类迁移难度 | 中 | 中 | 高 | 中高 | 低至中 |

# 三、信息流比较

## 3.1 mmFormer：可用模态流向共享 token

```text
各可用 MRI → 独立 encoder/intra-modal token
                    ↓
缺失分支清零，观察 token 拼接
                    ↓
inter-modal self-attention
                    ↓
共享 bottleneck + 多尺度可用 skips → prediction
```

没有信息“流入缺失模态分支”；信息只流向一个共享表示。因此它属于**observed-modality fusion**，不是 completion。

## 3.2 M3AE：完整训练信号流向可恢复共享空间

```text
完整病例 → 随机整模态/patch 遮蔽 + x_sub 替代
                         ↓
masked reconstruction 学习跨模态冗余
                         ↓
同一患者两种缺失视图 → bottleneck self-distillation
                         ↓
缺失输入 + 冻结 x_sub → shared representation → prediction
```

患者信息通过可用 patch 影响重建，但 `x_sub` 本身是 dataset-level prior。它属于**替代表示 + 共享空间对齐**，而非患者特异生成。

## 3.3 ReHyDIL：当前域与历史域通过跨患者超图和队列双向连接

```text
当前模态患者 + replay 旧模态患者
              ↓ 同一 batch 的深层空间特征
动态跨患者超边 → HGNN propagation → segmentation
              ↓
旧模型预测队列 ↔ 当前预测队列 → TAC consistency
```

信息不流向一个“缺失模态 token”，而是在不同患者、不同模态域的 feature vertices 间传播。它是五篇中唯一的**跨患者高阶传播**，同时也是最容易产生 batch composition 依赖的方案。

## 3.4 D3Seg：先重组观察依赖，再定向生成缺失关键 latent

```text
可用模态 features → mask-aware multi-hop modality graph
                           ↓ condition
缺失 T1ce/FLAIR noise → latent diffusion → reliability re-masking
                           ↓
真实 + 生成 features → Transformer/Mamba fusion → decoder → EGDR
```

这是最明确的**显式潜特征补全**。图传播本身不产生缺失节点，扩散器才把可用信息定向转成缺失模态代理。

## 3.5 AdaMM：完整 teacher 知识流向缺失 student，而非流向缺失模态

```text
x_full → teacher feature/probability ─────────────┐
                                                  ↓ distillation
x_masked → adapter → student CF/GF anchor graphs → shared decoder → prediction
```

teacher 的信息通过 probability MSE、GSME、feature/adversarial alignment 流入 student。GARM 在 student 内部让组合特异与共享特征互相传播。它属于**教师知识迁移 + 关系细化**，不是 modality completion。

# 四、对我的方法最有价值的设计

## 4.1 Mask 显式进入融合与消息归一化

- **灵感来源：** mmFormer、D3Seg。
- **原论文实现：** mmFormer 屏蔽缺失 feature/token；D3Seg 删除模态图中缺失节点的行列并重算多跳邻接。
- **分类改造：** 将 `m∈{0,1}^4` 输入 ROI encoder、hyperedge gate 和 graph readout，而不是只把缺失体积置零。
- **融入超图：** anatomy incidence `H_prior` 固定；若某节点-模态观察缺失，对相应消息置零并按有效成员数重归一化。
- **预期收益：** 防止不同缺失数量造成 feature scale 漂移，并让模型明确区分“未观察”与“观察值接近零”。
- **风险：** 网络可能仅记忆 15 个 mask 模式；需要检查跨组合泛化。
- **必要消融：** zero fill；zero fill + mask；mask-aware node update；mask-aware hyperedge normalization。

## 4.2 单模态可用性与跨模态融合分阶段学习

- **灵感来源：** mmFormer 的 modality-specific encoders 与 auxiliary supervision。
- **原论文实现：** 每个模态独立编码并受单模态辅助分割监督，再进行 inter-modal Transformer。
- **分类改造：** 每个模态/ROI 分支先产生可分类 embedding，再进入局部 modality hyperedge 聚合。
- **融入超图：** 第一阶段在同一 ROI 内连接可用模态节点；第二阶段把 ROI embedding 放入 anatomy hypergraph。
- **预期收益：** 降低 T1ce 强模态垄断，保证单模态压力情形仍有信息。
- **风险：** 辅助头过强会妨碍互补融合。
- **必要消融：** 无辅助头、所有模态等权辅助、按缺失频率加权辅助；检查单模态与全模态 trade-off。

## 4.3 高比例遮蔽预训练与低成本缺失 token

- **灵感来源：** M3AE。
- **原论文实现：** 87.5% modality+patch mask 的 masked autoencoder，加 dataset-level `x_sub`。
- **分类改造：** 对 ROI token 做 modality masking 和 ROI masking，重建 teacher ROI embedding或低维统计，而非完整 MRI。
- **融入超图：** 为每种 `(ROI, modality)` 设置共享/条件 missing token；由同一 anatomy hyperedge 的观察成员提供上下文。
- **预期收益：** 在分类 fine-tuning 前学习跨 ROI、跨模态冗余，成本低于图像级生成。
- **风险：** 全局 token 可能退化成类别/数据集均值，造成虚假确定性。
- **必要消融：** zero token、global token、ROI-specific token、context-conditioned token；测 completion error 与分类增益相关性。

## 4.4 静态医学拓扑 + 缺失感知动态超边权

- **灵感来源：** D3Seg 的组合依赖图；同时吸收 ReHyDIL 动态关系的风险教训。
- **原论文实现：** D3Seg 从当前 feature 动态算四模态图；ReHyDIL 动态 KNN 直接改变跨患者超边。
- **分类改造：** 不让缺失 noisy feature 决定是否建立医学关系，只预测已有 anatomy hyperedge 的患者特异权重。
- **融入超图：** `H=H_prior` 固定，`w_e=f({x_v,m_v}_{v∈e})` 动态；可另加少量受约束 residual edges。
- **预期收益：** 同时获得可解释稳定性与个体适应性，避免全动态图拓扑漂移。
- **风险：** 错误先验会限制模型；动态权重仍可能被强模态控制。
- **必要消融：** static only、dynamic topology only、static+dynamic weights、static+unconstrained residual edges；报告边权稳定性。

## 4.5 分层 teacher-student relational distillation

- **灵感来源：** AdaMM，辅以 M3AE self-distillation。
- **原论文实现：** full teacher 与 missing student 对齐 probability、bottleneck 和 feature relations；M3AE 对同一患者两种缺失视图对齐 bottleneck。
- **分类改造：** 完整 teacher 监督缺失 student 的 logits、ROI nodes、hyperedge embeddings 和 graph readout；两个随机 mask 的 student 也做一致性。
- **融入超图：** 对齐的是超边内关系或 edge response，而非要求每个缺失 node 与 teacher 数值完全相同。
- **预期收益：** 将完整模态结构知识迁移到缺失输入，并保持患者特异关系。
- **风险：** teacher error、过度对齐和 loss 过多；天然不完整训练病例无法提供 full pair。
- **必要消融：** logit KD、node KD、hyperedge relational KD、student-student consistency 单独与组合；验证 teacher 上界。

## 4.6 组合式适配而非 14 个固定 adapter

- **灵感来源：** AdaMM 的 combination-aware adapter。
- **原论文实现：** 四模态的 14 个非完整组合各自有 adapter。
- **分类改造：** 每个模态学习一个 residual expert，或以 mask encoder/hypernetwork 生成 FiLM/gating 参数，组合出当前适配。
- **融入超图：** adapter 作用于 modality-specific ROI nodes 或 hyperedge gate，不复制完整 HGNN。
- **预期收益：** 保留组合条件性，同时避免 `2^M` 参数增长。
- **风险：** 可组合性假设可能忽略“同时缺 T1ce+FLAIR”的非加性效应。
- **必要消融：** fully shared、14 adapters、per-modality additive adapters、mask-hypernetwork；比较参数量和 worst-pattern 指标。

## 4.7 跨患者高阶关系只在受控 memory bank 中使用

- **灵感来源：** ReHyDIL。
- **原论文实现：** batch 中患者-空间 vertices 经动态 KNN 超边传播。
- **分类改造：** 每个患者是一个 graph-level node，训练集 memory bank 中同等级/影像表型/临床先验近邻构成超边；查询患者只能读取训练 memory。
- **融入超图：** 患者内 ROI anatomy hypergraph 先编码，再以 graph embedding 构建第二层 cross-patient hypergraph。
- **预期收益：** 利用队列中的高阶群体表型，尤其可能帮助极端缺失模式。
- **风险：** split leakage、test-batch依赖、标签泄漏、部署时 memory drift，风险高于患者内图。
- **必要消融：** no memory、train-only frozen memory、batch graph、label-free KNN；严格验证 memory 只含训练患者并测试 batch-size/order 不变性。

## 4.8 用可靠性门控替代无条件补全

- **灵感来源：** D3Seg context-robust re-masking。
- **原论文实现：** 比较不同条件下的扩散估计，低可靠 token 重新加噪生成。
- **分类改造：** 对每个补全 ROI token 估计 uncertainty，低可信时降低其节点/超边消息权，而非强制当作真实观察。
- **融入超图：** `message_{v→e}` 同时乘 observation mask 或 completion confidence；graph readout保留证据不确定性。
- **预期收益：** 避免错误 T1ce 表征通过超边扩散到所有 ROI。
- **风险：** 置信度可能未校准，网络可通过始终低权规避学习。
- **必要消融：** 无补全、无条件补全、补全+learned gate、补全+calibrated uncertainty；报告 ECE/Brier 与 worst-pattern accuracy。

# 五、可形成的候选方法方案

## 5.1 方案 A：稳定的静态 Anatomy Hypergraph

### 核心假设

缺失模态分类首先需要稳定、可解释的患者内关系；在数据有限时，正确的 anatomy prior 比高度动态的图或生成器更可靠。

### 节点和超边定义

- 节点：固定 ROI 的融合表示，每个 ROI 内先保留四个 modality-specific tokens 和 observation mask。
- 局部超边：同一 ROI 中所有可用模态 token 构成 modality hyperedge。
- 全局超边：按解剖系统、肿瘤邻域或临床定义连接多个 ROI，incidence matrix 预先固定。

### 缺失模态处理方式

不做显式补全。缺失 token 不发消息；hyperedge aggregation 按有效成员重归一化。使用可学习 missing embedding 仅标记缺失，不冒充患者影像。

### 主干结构

模态特异 3D ROI encoder → ROI 内 mask-aware hyperedge aggregation → 2-3 层静态 HGNN → attention/readout classifier。

### 融合方式

借鉴 mmFormer 的“先模态内、后跨模态”，但把跨模态 attention 改为可解释的 ROI-local hyperedge；跨 ROI 只走医学先验超边。

### 损失函数

主分类 CE/focal loss + 单模态/单 ROI 辅助分类 + 两种随机 mask 的 logit consistency。初版不加 teacher、重建或生成损失。

### 训练协议

每病例逐迭代均匀或临床先验加权采样 15 个非空 mask；full、缺 T1ce、单模态和随机缺失分层验证；同一 split/checkpoint criterion。

### 最小消融实验

1. concat/mean fusion vs local modality hyperedge。
2. 无 anatomy HGNN vs static anatomy HGNN。
3. zero fill vs mask-aware normalization。
4. 无辅助/一致性损失 vs 分别加入。

### 新增复杂度和风险

复杂度最低、解释性最高；风险是静态先验表达能力有限，若 ROI 提取误差大，固定超边会持续传播错误。

## 5.2 方案 B：静态拓扑 + 缺失感知动态边权

### 核心假设

医学关系的“存在性”应稳定，但不同患者和缺失组合下每条关系的可靠程度应变化；动态更新权重足以获得适应性，无需完全重建拓扑。

### 节点和超边定义

沿用方案 A 的 `(ROI, modality)` 局部节点和 anatomy ROI 超边。另定义 modality hyperedges、左右半球/肿瘤核心-周边等先验边族。

### 缺失模态处理方式

mask-conditioned edge network 根据可用成员、缺失数量和 ROI feature 预测 `w_e`；可选轻量 context-conditioned token 补偿缺失 ROI-modality 表征，并输出 confidence。

### 主干结构

模态特异 ROI encoder → static incidence `H_prior` → edge-weight network → weighted HGNN → gated graph readout → classifier。

### 融合方式

借鉴 D3Seg 的组合依赖重估，但仅改变超边权；借鉴 M3AE 的 missing token，但不做高成本体积重建。补全消息乘 confidence 后才进入超边。

### 损失函数

分类 loss + two-mask consistency + 可选 ROI embedding reconstruction + edge-weight sparsity/stability regularization。避免一开始加入扩散。

### 训练协议

先在完整/轻缺失条件训练 encoder 与静态图，再逐步增加缺失强度；最后联合均衡采样 15 组合。验证动态权重在相同患者不同 mask 下是否合理稳定。

### 最小消融实验

1. static equal weight vs learned global weight vs patient/mask dynamic weight。
2. dynamic weights vs unconstrained dynamic KNN topology。
3. no completion vs missing token vs context completion。
4. no confidence vs confidence gating。

### 新增复杂度和风险

比方案 A 增加 edge network 和可能的 completion head，但远低于扩散；主要风险是动态权重学成 mask 查表或被 T1ce 单一模态主导。

## 5.3 方案 C：跨患者超图 + Teacher-Student Consistency

### 核心假设

极端缺失输入的单患者证据不足时，可从训练患者的群体级高阶表型获得帮助；完整 teacher 可约束这种传播不偏离完整模态决策。

### 节点和超边定义

- 第一级：患者内 ROI anatomy hypergraph，与方案 B 相同。
- 第二级：每个患者的 graph embedding 作为节点；在仅含训练患者的 memory bank 中按影像表型/无标签 embedding KNN 形成 cross-patient hyperedges。
- 禁止用测试标签、验证患者或同批测试患者共同构图。

### 缺失模态处理方式

missing student 先由患者内静态+动态权重超图编码，再读取 frozen/train-only patient memory；不生成 MRI。teacher 使用完整四模态但共享 ROI/超图语义。

### 主干结构

full-modal teacher HGNN + missing-modal student HGNN；student 后接受控 cross-patient HGNN/memory attention，最终分类。

### 融合方式

患者内融合产生 graph embedding；患者间超边仅补充群体表型。teacher 蒸馏 node、hyperedge、graph embedding 和 logits，借鉴 AdaMM 但不使用 14 个固定 adapters。

### 损失函数

主分类 loss + temperature-scaled logit KD + hyperedge relational KD + two-mask student consistency + memory contrastive loss。逐项加入，不一次启用全部。

### 训练协议

先训练 full teacher；冻结 teacher 后训练 missing student；memory bank 每折只由该折训练患者构建，validation/test 只作为 query。报告不同 batch size、query order 和 memory size 的稳定性。

### 最小消融实验

1. student only vs +logit KD vs +hyperedge KD。
2. no patient memory vs train-only memory。
3. pairwise KNN graph vs true multi-patient hyperedge。
4. batch graph vs frozen memory，专门检查泄漏与 batch dependence。
5. teacher full-modal upper bound 与 student worst-mask gap。

### 新增复杂度和风险

训练和验证复杂度最高。最大风险不是计算量，而是 split leakage、transductive test dependence、teacher bias 和方法故事过重。只有方案 A/B 已建立可靠提升、且严格 train-only memory 仍有额外收益时才值得采用。

## 5.4 三个方案的决策建议

优先顺序应为 **A → B → C**。方案 A 建立可解释、协议干净的分类基线；方案 B 检验“动态性是否真的需要”；方案 C 只有在极端缺失仍存在明确性能缺口时再引入。这样每一步都对应一个可证伪假设，而不是把五篇论文的所有模块堆到同一网络。

# 六、阅读顺序建议

## 1. ReHyDIL

先读它不是因为最容易迁移，而是因为它是与你“超图 + 缺失脑 MRI”最接近、也最需要准确区分的工作。重点读：DIL 问题定义、CPH 节点/超边构造、HGNN 传播、TAC 和 replay protocol、Table 2 消融。阅读后必须能说明：为什么它是真超图、为什么其 cross-patient setting 不等于你的 within-patient ROI 分类。

## 2. mmFormer

第二篇用于建立最清楚的任意缺失组合数据流基线。重点读：四个 modality-specific encoders、mask 在 feature 级如何应用、intra/inter-modal Transformer、辅助监督、15 组合 protocol。阅读后应能把它转写成你的“先 ROI 内模态聚合、后 ROI 间关系推理”。

## 3. AdaMM

第三篇直接回答怎样利用完整四模态训练信息而不增加推理负担。重点读：teacher/student 输入配对、BBDM、GARM 的普通图性质、组合 adapter 数量、LGRM 风险。代码重点核对 `AdaMMKD.py` 和 `AdaGEM.py`，同时记住当前仓库并不完整。

## 4. M3AE

第四篇用于决定是否需要预训练和缺失 token。重点读：87.5% masking、`x_sub` 的 model inversion、pretrain/fine-tune 分界、two-view self-distillation。最重要的判断是：`x_sub` 是全局替代先验，不是患者特异缺失 MRI；迁移时更适合 ROI token reconstruction。

## 5. D3Seg

最后读 D3Seg，因为它最复杂、发表状态和实现证据最不稳定。重点读：MMGF 的普通多跳图、只补 T1ce/FLAIR 的 latent diffusion、re-masking、推理成本和 EGDR 的 segmentation specificity。阅读目的不是直接采用扩散，而是判断“选择性补全 + 可靠性门控”是否在简单 completion baseline 失败后值得增加。

**最终应形成的主线：** ReHyDIL 定义你的最近超图边界，mmFormer给出稳健 mask/fusion 基线，AdaMM给出 full-to-missing knowledge transfer，M3AE给出低成本缺失表示预训练，D3Seg作为高复杂度显式补全上限。对当前课题，最合理的第一方法不是五者相加，而是“mask-aware ROI 编码 + 静态 anatomy hypergraph + 可验证的动态边权”，再按证据逐步加入 relational distillation 或低置信 completion。
