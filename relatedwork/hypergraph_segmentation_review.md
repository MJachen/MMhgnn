# 超图学习与医学影像分割：面向缺失模态脑 MRI 分类的设计启发（2021—2026）

> 核查日期：2026-07-14  
> 研究目标：不是建立通用 Related Work，而是判断“分割监督怎样以最低风险进入现有 5 节点、5 固定超边的 anatomy-only hypergraph classifier”。  
> 证据标记：**[论文事实]** 来自论文正文/补充材料；**[代码事实]** 来自官方仓库；**[分析推断]** 是针对当前网络的迁移判断，不代表原作者结论。

## 检索与筛选说明

检索覆盖 2021—2026 年，核心查询包括 `hypergraph neural network medical image segmentation`、`graph reasoning brain tumor segmentation`、`segmentation-guided glioma classification`、`missing modality brain tumor segmentation`、`teacher student missing modality segmentation` 等组合。优先核查论文 PDF、会议/期刊元数据、补充材料和官方代码；arXiv 工作单独标为预印本。纳入标准是：真实使用医学影像超图/图分割，或能为 anatomy ROI、分割辅助分类、缺失模态监督提供明确可迁移机制。排除了仅名称含 hypergraph、无可核实方法、纯自然图像方法及与 ROI/辅助分割无实质联系的工作。

最终保留 18 篇：4 篇真实 hypergraph、5 篇 ordinary graph、3 篇 graph-like relational reasoning、6 篇无图但与多任务/缺失模态协议直接相关的参考。这个分布本身说明：**近五年“超图 + 医学分割”仍是小样本文献群，不能把 Transformer attention、多跳普通图或模态分组误写成超图。** 完整逐项矩阵见 [hypergraph_segmentation_papers.csv](./hypergraph_segmentation_papers.csv)。

---

## 1. Executive Summary

### 1.1 近五年的主要技术路线

1. **像素/patch 动态超图嵌入 encoder-decoder。** AHGNN 在低分辨率特征层把空间位置作为节点，以特征 KNN 超边叠加局部滑窗超边；UHR-Net 则用前景/背景超边原型和 soft incidence 对粗分割进行不确定性感知细化。
2. **跨模态或跨患者高阶关系学习。** HyCon 把跨图像、跨模态特征组织为超边进行对比学习和蒸馏；ReHyDIL 把批次中的患者-空间位置共同建成动态超图，用于缺失模态域增量分割。
3. **普通区域图/模态图辅助分割。** GNN-Tumor-Seg 使用 supervoxel 邻接图完成 region prediction 后回投到 voxel grid；AdaMM、D3Seg 分别使用原型图和模态图处理缺失模态。它们有明确二元边，**不是超图**。
4. **关系监督而非显式推理图。** Graph Flow 蒸馏跨层通道关系；mmFormer、GSS 用 attention 或类别级模态一致性组织信息，适合作为训练机制参考，但没有 incidence matrix。
5. **分割辅助分类/预测。** 共享 encoder 双头、分割后 ROI 引导分类、分割特征与分类特征交互、分割预训练/正则化、完整模态教师向缺失模态学生传递分割知识，是五种主要范式。

### 1.2 超图在分割中通常建模什么

超图的实际价值不在“节点更多”，而在一条超边同时表达多个实体的共同关系：局部形态邻域、远距离语义相似区域、跨模态同源区域、前景/背景原型归属、跨患者同类结构。对于当前网络，最自然的高阶实体已经存在：5 个 anatomy ROI 与 5 条固定医学先验超边。因此最有价值的迁移不是重建大规模 voxel/patch KNN，而是让分割监督改善 **节点的空间支持集、节点置信度和固定超边的样本级权重**。

### 1.3 分割为何能帮助分类

分割可通过三条路径帮助分类：第一，迫使共享 encoder 保留肿瘤位置、亚区和边界信息；第二，用预测概率而非粗 ROI 生成更语义一致的 node pooling masks；第三，以分割置信度估计 node/hyperedge reliability，在模态缺失时降低不可靠区域的传播。风险是像素级损失梯度通常更密集、更大，可能把共享表征推向“边界可分”而不是“分型可分”，产生负迁移。

### 1.4 最适合当前网络的设计

最先实现的方案应是：**保留现有分类路径、5 节点和 5 固定超边完全不变，只在各模态 encoder 的共享语义特征后增加轻量辅助分割头；训练时使用相同 modality dropout，推理时丢弃分割头。** 先证明辅助监督本身能提升 validation balanced accuracy 和 15 组合稳健性，再进入 soft anatomy nodes；超边置信度门控应放在第三阶段。

### 1.5 暂不适合直接采用的设计

- voxel/supervoxel 大图：与当前仅 5 个解剖节点的低成本优势冲突。
- sample-wise KNN 动态拓扑：缺失模态导致距离空间漂移，可能恢复此前应避免的不稳定性。
- 14 个缺失模式 adapter 或 15 个专用网络：部署和维护成本过高。
- diffusion modality completion：难以单独归因分割辅助是否有效，且不是当前 MVP 所需。
- hard segmentation cascade：一次错误会直接污染 ROI 节点；测试时也不应依赖真实分割标签。

---

## 2. Representative Works

### 分级总览

| 分级 | 论文 | 主要阅读目的 |
|---|---|---|
| A. 必须精读 | AHGNN; UHR-Net; ReHyDIL; GNN-Tumor-Seg; AdaMM; GliomaNet; Multi-task IDH/1p19q; mmFormer; M3AE | 直接回答超图位置、soft incidence、缺失模态协议、分割到分类的连接方式 |
| B. 方法启发 | HyCon; ViG-UNet; GraphCL; D3Seg; ACN; GSS; modality-agnostic meta-learning | 跨模态一致性、teacher-student、动态图风险、自然缺失训练 |
| C. 背景参考 | Graph Flow; breast mpMRI segmentation-to-classification cascade | 关系蒸馏与 hard ROI 级联的可取点和风险 |

以下逐篇分析不重复堆叠 Dice 数值，而集中于方法位置、节点/边定义、训练依赖和迁移条件。

### A. 必须精读

## A Novel Adaptive Hypergraph Neural Network for Enhancing Medical Image Segmentation

### 1. 基本信息

- 年份/来源：2024，MICCAI，正式发表；[论文与补充材料](https://papers.miccai.org/miccai-2024/027-Paper2689.html)。
- 任务/数据：Synapse 腹部 CT 多器官分割、ACDC 心脏 MRI 分割；非脑肿瘤。
- 代码：[AHGNN 官方仓库](https://github.com/11yxk/AHGNN)，已公开。

### 2. 论文解决的问题

**[论文事实]** CNN 局部卷积难以显式表示远距离高阶相似区域；固定 K 的图/超图又不能适应不同节点密度。作者因此把局部空间关系和非局部语义关系放入同一超图模块。论文不研究多模态缺失。

### 3. 方法的整体思想

一句话：**通过自适应特征 KNN 超边与局部滑窗超边共同建模空间位置之间的高阶关系，从而在低分辨率语义层增强分割特征。**

```text
2D image → CNN encoder → low-resolution feature nodes
         → adaptive KNN hyperedges + local-window hyperedges
         → hypergraph convolution + CNN/residual fusion
         → decoder with deep supervision → segmentation
```

### 4. 节点、边和超边定义

节点是低分辨率特征图展平后的空间位置/patch。每个节点以特征空间邻居组成一条多节点超边，邻居规模由节点度调节；另有局部滑窗超边保持空间连续性。incidence matrix 来自上述成员关系。结构为患者内、样本级动态 KNN 加固定局部邻域。

### 5. 超图如何嵌入分割网络

AHGNN 放在低分辨率 encoder/decoder 语义层，与卷积分支并行并残差融合；输出仍保持规则网格形状，因而不需要复杂 graph-to-grid 重建。它强调多尺度深监督，而非直接在原分辨率 voxel 上建图。

### 6. 损失函数和监督信号

**[论文事实]** 使用 `0.3 CE + 0.7 Dice`，decoder 多尺度 deep supervision；未报告独立的超边标签或拓扑正则。

### 7. 方法优势

同时保留局部连续性和远距离高阶语义；超边包含多个相似位置，表达能力强于仅连接两个 patch 的普通图；低分辨率推理控制了计算量。

### 8. 局限和风险

动态 KNN 特征距离会受缺失模态影响；节点/超边没有医学解剖语义；数据集中没有脑肿瘤或任意模态缺失；高阶增益与额外参数/局部分支的贡献未完全解耦。

### 9. 对我的方法的启发

- **可以直接借鉴：** 超图放在低分辨率语义层；超图更新使用残差；局部形态与远程语义分开编码。
- **需要改造后借鉴：** 把 patch 节点改为现有 5 个 anatomy ROI；保留固定 incidence matrix，只让 segmentation confidence 调节边权。
- **不建议直接借鉴：** 在每个受试者上重新做 sample-wise KNN 或把 voxel 作为节点。

### 10. 阅读优先级

**必须精读；重点看方法图、adaptive hyperedge construction 和 residual fusion 公式。**

## UHR-Net: An Uncertainty-Aware Hypergraph Refinement Network for Medical Image Segmentation

### 1. 基本信息

- 年份/来源：2026，[arXiv:2604.28095](https://arxiv.org/abs/2604.28095)，截至核查日为预印本，未发现正式发表版本。
- 任务/数据：息肉、腺体、皮肤病灶分割；Kvasir-SEG、Kvasir-Sessile、GlaS、ISIC 2016/2017。
- 代码：[UHR-Net 仓库](https://github.com/CUGfreshman/UHR-Net)，公开但较新、无 release。

### 2. 论文解决的问题

**[论文事实]** 粗分割最不可靠的位置通常在边界与模糊前景；硬 KNN 或固定超边不能表达节点归属不确定性。方法因此让 coarse posterior、前景/背景原型和预测熵共同决定 soft hypergraph。

### 3. 方法的整体思想

一句话：**利用粗分割上下文生成样本特异的前景/背景超边原型，并用预测熵调节 soft incidence，以高阶传播细化不确定边界。**

```text
image → encoder/coarse prediction → pixel nodes
      → foreground/background hyperedge prototypes + sample offsets
      → entropy-modulated soft incidence
      → node↔hyperedge propagation || dilated convolution
      → residual multi-scale decoder → refined segmentation
```

### 4. 节点、边和超边定义

节点是特征图的像素位置。默认前景和背景各 8 个超边原型；粗分割上下文产生原型偏移，节点与原型点积形成归属 logits，熵参与调节，再沿节点维归一化得到 soft incidence。一条超边可包含所有节点的不同权重，因此不是普通 KNN 图。

### 5. 超图如何嵌入分割网络

UGHR blocks 位于多尺度 decoder；节点→超边聚合后再回到节点，与并行 dilated convolution 分支残差融合。低/中分辨率执行高阶推理，输出直接 reshape 回网格。

### 6. 损失函数和监督信号

**[论文事实]** Dice + BCE；另用 uncertainty-oriented image composition 进行预训练，通过病灶 copy-paste 构造正样本和病灶样背景 hard negatives。没有真实超边标签。

### 7. 方法优势

把 segmentation uncertainty 变成结构变量而非仅报告指标；soft incidence 比 hard ROI 更能容忍边界模糊；前景/背景原型使超边具有可解释语义。

### 8. 局限和风险

仅预印本；主要是 2D 非 MRI 数据；像素级 soft incidence 仍有计算成本；粗分割错误可能改变原型和传播；代码成熟度有限。

### 9. 对我的方法的启发

- **可以直接借鉴：** 用分割熵估计 boundary node confidence；以 residual gate 限制错误门控幅度。
- **需要改造后借鉴：** 将像素节点/前景背景原型改为 5 个 anatomy nodes；soft incidence 不重建拓扑，只生成 5 条固定超边的权重。
- **不建议直接借鉴：** 在 3D 体素层完整复现多尺度 soft hypergraph，或让低置信分割完全关闭某条医学先验超边。

### 10. 阅读优先级

**必须精读；重点看 soft incidence、entropy modulation、residual refinement。结论需按预印本证据等级使用。**

## ReHyDIL: Resilient Hypergraph Neural Networks for Domain-Incremental Learning with Missing Modalities in Medical Image Segmentation

### 1. 基本信息

- 年份/来源：2025，MICCAI，正式发表；[论文页](https://papers.miccai.org/miccai-2025/0430-Paper2774.html)。
- 任务/数据：BraTS 2019，335 例，缺失模态域增量脑肿瘤分割。
- 代码：[ReHyDIL](https://github.com/reeive/ReHyDIL)，已公开。

### 2. 论文解决的问题

**[论文事实]** 模态以域序列到达时，网络既要学习新模态域又要避免遗忘旧域。作者认为深层语义位置之间存在可复用的高阶关系，因此把动态超图与 replay、拓扑感知对比学习结合。

### 3. 方法的整体思想

一句话：**在深层 encoder features 上构建跨患者动态超图，并通过回放和 topology-aware contrastive queue 保持缺失模态域增量过程中的结构知识。**

```text
current-domain MRI + replay samples → segmentation encoder
    → layer-4/5 patient-spatial nodes
    → adaptive KNN + local hyperedges → CHSNet
    → decoder segmentation
    ↘ topology-aware contrastive queue / continual-learning regularization
```

### 4. 节点、边和超边定义

encoder 第 4/5 层 `B×C×H×W` 的位置展平为 `N=BHW` 节点，因而图跨患者和空间位置。每个节点由自适应特征 KNN 形成动态超边，再叠加局部滑窗超边。结构随批次和当前域变化。

### 5. 超图如何嵌入分割网络

CHSNet 位于深层语义层，输出回到 decoder；不是在原始 3D voxel 上建图。replay memory 和对比 queue 仅用于训练，分割输出仍由常规 decoder 恢复到图像空间。

### 6. 损失函数和监督信号

分割损失与 topology-aware contrastive/continual-learning objectives 联合；使用 10% replay。训练为 2D axial 224×224、200 epochs、batch 64、Adam `lr=1e-4, wd=4e-4`。

### 7. 方法优势

这是检索集中最接近“hypergraph + brain tumor + missing modality”的正式论文；它证明深层高阶关系可作为跨域稳定知识，而非仅作为分割 decoder 的局部插件。

### 8. 局限和风险

模态顺序是 `T1→T2→FLAIR→T1ce` 的 domain-incremental protocol，不等同于一个模型覆盖任意 15 种组合；batch-level graph 可能依赖批次组成；动态 KNN 可能在模态变化时漂移。

### 9. 对我的方法的启发

- **可以直接借鉴：** 在深层语义层施加拓扑/节点一致性；少量 replay 可稳定跨组合训练。
- **需要改造后借鉴：** 把跨患者动态图改为同一受试者的固定 5 节点 anatomy hypergraph；对比目标比较同一受试者不同缺失组合下的 node embeddings。
- **不建议直接借鉴：** 把顺序域增量结果直接当作 15 组合缺失鲁棒性证据，或恢复跨患者 KNN 推理。

### 10. 阅读优先级

**必须精读；重点核对任务设定、CHSNet 节点维度和 replay/TAC 协议，避免问题设定错配。**

## A Joint Graph and Image Convolution Network for Automatic Brain Tumor Segmentation

### 1. 基本信息

- 年份/来源：BrainLes 2021 workshop，LNCS 正式卷于 2022 出版；[论文](https://arxiv.org/abs/2109.05580)。
- 任务/数据：BraTS 2021，T1/T1ce/T2/FLAIR 脑肿瘤分割。
- 代码：[GNN-Tumor-Seg](https://github.com/rsinghlab/GNN-Tumor-Seg)，公开并含完整 pipeline。

### 2. 论文解决的问题

**[论文事实]** 纯 CNN 在固定网格上计算，而肿瘤区域形状和强度分布更适合 region-level 表示。作者先用 supervoxel 图产生区域预测，再由浅 CNN 恢复细节。该方法是 ordinary graph，不是 hypergraph。

### 3. 方法的整体思想

一句话：**将四模态 MRI 压缩为 supervoxel adjacency graph 完成区域级分割，再把 region logits 回投到 voxel grid 并用 CNN 细化。**

```text
4-modal MRI → supervoxels (~15k) → node intensity features
            → adjacency GraphSAGE → region logits
            → graph-to-grid projection → shallow CNN refinement
            → voxel segmentation
```

### 4. 节点、边和超边定义

节点是患者内 supervoxels，节点特征汇总四模态强度；边连接空间相邻 supervoxels，仅连接两个节点。图由预处理固定，患者内静态。一条“超边包含几个节点”不适用，因为它没有超边。

### 5. 图如何嵌入分割网络

7 层 GraphSAGE-pool 先输出 region labels；根据 supervoxel assignment 映射回 voxel grid；2 层 CNN 仅负责细化。这是最清晰的 region-to-voxel 桥接案例之一。

### 6. 损失函数和监督信号

区域标签由 voxel mask 聚合得到；GNN 超参以 random search + 5-fold CV 选择，最佳设置在全部数据上训练 3 次并按验证集选择。损失细节应以 PDF/代码为准。

### 7. 方法优势

region 节点具有比 patch 更强的空间语义；图到网格映射明确；先全局关系、后局部细化的分工清晰。

### 8. 局限和风险

约 15,000 节点仍然昂贵；supervoxel 错误构成上限；默认完整四模态；普通邻接图无法直接表达 core-boundary-peritumoral 的多节点医学先验。

### 9. 对我的方法的启发

- **可以直接借鉴：** 明确保存 segmentation posterior 与 anatomy ROI 之间的双向映射；region-level 表示比 voxel graph 更符合当前设计。
- **需要改造后借鉴：** supervoxel assignment 改为 5 个 soft anatomy masks，region logits 改为节点置信度。
- **不建议直接借鉴：** 在当前分类器前增加 15k 节点 GraphSAGE，或以 hard supervoxel label 替代已有 ROI。

### 10. 阅读优先级

**必须精读；重点看 supervoxel node construction、graph-to-grid projection 和 CNN refinement。**

## AdaMM: Adaptive Missing-Modality Brain Tumor Segmentation with Learnable Graph Reasoning and Diffusion-based Modality Synthesis

### 1. 基本信息

- 年份/来源：2026，Medical Image Analysis 112:104108，正式发表；[DOI](https://doi.org/10.1016/j.media.2026.104108)。
- 任务/数据：BraTS 2018、BraTS 2024、Pretreat-MetsToBrain-Masks；任意缺失模态脑肿瘤/转移瘤分割。
- 代码：[AdaMM](https://github.com/Quanato607/AdaMM) 已公开，但核查快照缺少部分 README 所述入口/模块，应标为“部分可复现”。

### 2. 论文解决的问题

**[论文事实]** 缺失模态同时造成内容缺失和模态关系改变；单纯合成图像不能保证结构关系正确。AdaMM 使用完整模态教师、缺失模态学生、扩散补全与双图关系推理共同解决。它使用普通图，不是超图。

### 3. 方法的整体思想

一句话：**以完整/生成特征形成双原型图，通过跨图关系推理对齐缺失表示，同时用扩散补全和病灶存在性辅助目标改善分割。**

```text
full teacher features ─→ K=8 anchor graph ┐
missing student + synthesis → K=8 graph ├→ cross-graph GAT → decoder
availability mask / lesion-presence head ┘              → segmentation
```

### 4. 节点、边和超边定义

每个患者 K=8 个可学习 anchor prototypes 为节点；complete-feature graph 与 generated-feature graph 内部由 pairwise similarity 建边，跨图边由 cosine similarity threshold 0.8 建立，再经 GAT 推理。所有边均为二元边；无 incidence matrix。

### 5. 图如何嵌入分割网络

图模块位于深层表示/融合阶段，输出进入 segmentation decoder；图不直接产生 voxel mask。BBDM 补全与 GARM/LGRM 关系模块并行协作。

### 6. 损失函数和监督信号

联合 segmentation、diffusion/reconstruction、graph alignment 和 lesion-presence objectives。每次迭代随机选择 15 个非空 availability masks 之一并 zero-fill；教师分支使用完整模态。

### 7. 方法优势

显式区分“补全内容”与“恢复关系”；lesion-presence auxiliary classifier 为抑制假阳性提供额外全局监督；训练与全部缺失组合一致。

### 8. 局限和风险

14 个缺失模式 adapters 增加维护成本；完整模态教师假设不适用于所有现实数据；动态图由生成质量驱动；扩散模块使归因复杂；仓库快照不完整。

### 9. 对我的方法的启发

- **可以直接借鉴：** 同一病例完整/缺失组合间蒸馏节点和关系；availability mask 进入关系权重；病灶存在性可作为小权重辅助目标。
- **需要改造后借鉴：** 8 个 anchors 改为固定 5 个 anatomy nodes；双普通图对齐改为节点/超边响应一致性。
- **不建议直接借鉴：** 14 adapters、diffusion completion、动态跨图拓扑作为初始方案。

### 10. 阅读优先级

**必须精读；重点看 mask sampling、graph alignment 和辅助病灶存在性，代码可复现性需单独审计。**

## An Attention-Guided CNN Framework for Segmentation and Grading of Glioma Using 3D MRI Scans

### 1. 基本信息

- 年份/来源：2023，IEEE/ACM TCBB，正式发表；[PubMed](https://pubmed.ncbi.nlm.nih.gov/36350865/)。
- 任务：先分割胶质瘤，再预测 HGG/LGG、IDH、1p/19q。
- 代码：[GliomaNet](https://github.com/prasunc/Gliomanet)，公开；精确 cohort、split 和超参仍需人工核查。

### 2. 论文解决的问题

**[论文事实]** 全脑输入含大量与分级/分子状态无关的背景，先定位肿瘤有助于下游预测。该工作是 attention-guided segmentation 加 downstream multi-task grading 的两阶段 pipeline，不应写成 hypergraph，也不应在未核实共享梯度前写成端到端联合训练。

### 3. 方法的整体思想

一句话：**用空间/通道注意力分割网络定位 3D 肿瘤，再将预测 ROI 交给多终点 CNN 完成分级和分子状态预测。**

```text
multimodal 3D MRI → attention segmentation network → predicted tumor ROI
                  → grading/molecular CNN → HGG/LGG, IDH, 1p/19q
```

### 4. 节点、边和超边定义

无显式节点、边或超边。attention map 是连续权重，不构成普通图或超图。

### 5. 分割如何嵌入预测网络

分割位于分类前级，作用是 ROI generation，而不是在 hypergraph 中传播。测试阶段使用预测 mask；不能用真实测试 mask 代替。

### 6. 损失函数和监督信号

分割和多终点分类分别使用相应监督；具体联合/分阶段优化和权重需以论文、代码核对，当前不作猜测。

### 7. 方法优势

任务与本研究最接近：脑 MRI、肿瘤定位、分级/分子预测；说明 segmentation 的价值可能首先出现在 ROI 选择，而不仅是 encoder regularization。

### 8. 局限和风险

hard cascade 会传播分割错误；不研究缺失模态；若分割与分类不共享 encoder，则不能证明 dense supervision 改善分类特征；数据协议仍需人工核查。

### 9. 对我的方法的启发

- **可以直接借鉴：** segmentation posterior 参与 ROI pooling；多终点任务共享受试者表示。
- **需要改造后借鉴：** hard tumor crop 改为 soft probability masks，并与现有 anatomy ROI 做残差混合。
- **不建议直接借鉴：** 测试时使用真实 ROI，或让分割失败直接置空 core/boundary 节点。

### 10. 阅读优先级

**必须精读；重点看 segmentation-to-grading 数据接口和训练是否真正联合。**

## Multi-task glioma segmentation and IDH mutation and 1p19q codeletion classification via a deep learning model on multimodal MRI

### 1. 基本信息

- 年份/来源：2025，Meta-Radiology，正式发表；[论文页](https://www.sciencedirect.com/science/article/pii/S2950162825000207)。
- 任务/数据：BraTS 2020 中 132 例具有分子标签的检查；肿瘤亚区分割、IDH 和 1p/19q 分类。
- 代码：未发现官方仓库。

### 2. 论文解决的问题

**[论文事实]** 分割和分子预测可共享肿瘤表征，从有限分子标签中获得更稳定的空间特征。它是共享 encoder/bottleneck 的多任务参考，不包含 graph。

### 3. 方法的整体思想

一句话：**以 CNN encoder-decoder 预测肿瘤亚区，并在共享 bottleneck 上连接 IDH 与 1p/19q 分类 heads。**

```text
T1/T1ce/T2/FLAIR → shared CNN encoder → bottleneck
                                   ├→ decoder → tumor subregions
                                   ├→ IDH head
                                   └→ 1p/19q head
```

### 4. 节点、边和超边定义

没有显式节点、边或超边；bottleneck feature vector 不是图节点。

### 5. 分割如何嵌入网络

分割 decoder 与分类 heads 共享 encoder/bottleneck，属于“共享编码器 + 双任务头”。公开描述还包含各模态预测的 majority vote，但它不能自然表达任意缺失组合。

### 6. 损失函数和监督信号

联合使用亚区 segmentation labels 和两个分子分类 labels；公开预览未完整给出损失公式和任务权重，标记为需要人工核查。

### 7. 方法优势

结构简单、与“分类主任务 + 分割辅助任务”最接近；肿瘤亚区监督与 core/boundary/peritumoral 节点语义有直接联系。

### 8. 局限和风险

样本仅 132 例且类别不平衡，无外部验证；默认完整四模态；majority vote 忽略模态质量；未分析任务梯度冲突。

### 9. 对我的方法的启发

- **可以直接借鉴：** 在当前 modality encoders 后加轻量 decoder；分类 checkpoint 仍以 balanced accuracy 选择。
- **需要改造后借鉴：** majority vote 改为现有 mask-aware modality gate；分割设为辅助而非同等主任务。
- **不建议直接借鉴：** 以固定等权损失开始长时间联合训练，或只报告 segmentation 改善而不报告 15 个分类组合。

### 10. 阅读优先级

**必须精读；重点看共享层深度、任务权重及小样本 protocol。**

## mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation

### 1. 基本信息

- 年份/来源：2022，MICCAI，正式发表；[论文](https://doi.org/10.1007/978-3-031-16443-9_11)。
- 任务/数据：BraTS 2018，任意缺失模态脑肿瘤分割。
- 代码：[mmFormer](https://github.com/YaoZhang93/mmFormer)，已公开。

### 2. 论文解决的问题

**[论文事实]** 直接拼接模态会把缺失位置与真实零信号混淆，也不能保持各模态独有信息。mmFormer 使用独立 3D CNN encoders 和 intra/inter-modal Transformers，在一个模型中处理全部非空组合。attention 是隐式关系，不是 graph/hypergraph。

### 3. 方法的整体思想

一句话：**先在每个模态内部编码，再只在可用模态 tokens 之间进行跨模态 attention，并以多尺度辅助分割稳定各模态表征。**

```text
available modalities → four modality-specific 3D encoders
                     → intra-modal Transformer
                     → mask-aware inter-modal Transformer
                     → decoder + modality/deep auxiliary segmentation
```

### 4. 节点、边和超边定义

没有显式 incidence matrix。tokens 可视为 attention 单元，但不存在离散二元边或多节点超边，故矩阵中标为 graph-like relational reasoning。

### 5. 关系推理如何嵌入分割网络

intra/inter-modal Transformer 位于各模态编码与融合阶段；缺失 feature slots 置零并被 mask；decoder 输出 voxel segmentation，多尺度/单模态分支提供辅助监督。

### 6. 损失函数和监督信号

主 segmentation loss 加 modality-specific/deep supervision。训练时四个模态独立 Bernoulli(0.5) 采样，拒绝全空组合；单模型测试全部 15 种组合。原始训练病例具备完整模态，再模拟缺失。

### 7. 方法优势

训练协议与当前“单模型、15 组合测试”高度一致；独立 modality encoders 与当前架构同构；每模态辅助分割能防止某一强模态完全支配融合。

### 8. 局限和风险

模型较重；训练依赖完整病例；attention 不提供当前 5 条医学先验超边的解释；直接移植 decoder 会显著改变主干。

### 9. 对我的方法的启发

- **可以直接借鉴：** 对可用模态施加 segmentation supervision；相同 availability mask 同时驱动分割与分类；all-combination protocol。
- **需要改造后借鉴：** Transformer 融合改为现有 ROI-wise modality gate；decoder 压缩为轻量训练期 head。
- **不建议直接借鉴：** 用大型 inter-modal Transformer 替换当前融合和固定 anatomy hypergraph。

### 10. 阅读优先级

**必须精读；重点看 modality dropout、auxiliary losses 和 missing mask 的数据流。**

## M3AE: Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities

### 1. 基本信息

- 年份/来源：2023，AAAI，正式发表；[论文](https://ojs.aaai.org/index.php/AAAI/article/view/25253)。
- 任务/数据：BraTS 2018/2020 任意缺失模态分割。
- 代码：[M3AE](https://github.com/ccarliu/m3ae)，已公开。

### 2. 论文解决的问题

**[论文事实]** 缺失模态不仅需要补偿信息，还需要使同一患者在不同 availability patterns 下得到稳定表示。M3AE 先做 modality/patch masked autoencoding，再进行同患者缺失视图 self-distillation。

### 3. 方法的整体思想

一句话：**通过大比例 masked multimodal autoencoding 学习跨模态先验，再约束同一病例不同缺失视图的 bottleneck 表示一致。**

```text
Stage 1: full cases → mask modalities/patches (ratio 0.875)
                    → autoencode/reconstruct multimodal content
Stage 2: same case, two missing views → shared encoder
                    → bottleneck consistency → segmentation fine-tuning
```

### 4. 节点、边和超边定义

没有图。patch tokens 和 latent features 是 masked modeling 单元，不构成边或超边。

### 5. 分割如何嵌入网络

分割是预训练后的下游主任务；model inversion 提供 global substitute，self-distillation 施加在 bottleneck。没有 segmentation posterior 到 ROI 的显式映射。

### 6. 损失函数和监督信号

阶段一 reconstruction，阶段二 bottleneck MSE/consistency 与 segmentation loss；单模型覆盖 15 组合，训练用完整病例构造缺失视图。

### 7. 方法优势

把 robustness 作用在表征层而非只依赖 modality completion；同一受试者跨组合一致性与当前节点稳定性目标直接一致。

### 8. 局限和风险

预训练成本高；global substitute 可能忽略局部 ROI 差异；训练完整病例假设较强；没有说明分类与分割梯度冲突。

### 9. 对我的方法的启发

- **可以直接借鉴：** 同一受试者不同模态组合的 subject embedding 一致性。
- **需要改造后借鉴：** bottleneck consistency 下沉为 5 个对应 anatomy node embeddings 的一致性，按分割置信度加权。
- **不建议直接借鉴：** 初始实验同时加入重建器、model inversion 和分割 decoder，导致无法归因。

### 10. 阅读优先级

**必须精读；重点看两阶段协议和同病例缺失视图一致性。**

### B. 方法启发

## Multi-modal hypergraph contrastive learning for medical image segmentation

### 1. 基本信息

- 年份/来源：2025，Pattern Recognition 165:111544，正式发表；[论文页](https://www.sciencedirect.com/science/article/pii/S0031320325002043)。
- 任务/数据：少标注肺部 CT/PET-CT 与 BraTS 多模态分割。
- 代码：[HyCon](https://github.com/reeive/HyCon)，已公开。

### 2. 论文解决的问题

**[论文事实]** 少量标注下，跨图像、跨模态的共同语义难由普通 pairwise contrastive learning 充分利用。HyCon 用超边把多个跨模态顶点作为高阶正样本，再把教师关系蒸馏给 MLP 学生。它不是缺失模态方法。

### 3. 方法的整体思想

一句话：**先生成成对跨模态样本，再以 teacher hypergraph contrastive learning 组织跨模态高阶正样本，最后蒸馏到轻量学生。**

```text
paired modalities → cross-modal generation
                  → HCLN teacher hypergraph
                  → topology/context/relation distillation
                  → MLP student → segmentation
```

### 4. 节点、边和超边定义

顶点来自不同图像和模态的特征；同一超边连接多个跨图像/模态顶点。公开预览未完整说明其空间粒度，因此不能进一步猜测为 voxel 或 ROI。

### 5. 超图如何嵌入分割网络

超图主要位于训练期 teacher；推理由蒸馏后的 MLP/student 完成，降低图推理开销。

### 6. 损失函数和监督信号

分割监督、hypergraph contrastive 和多级蒸馏共同优化；报告 5%/10% 标注设置。epoch、batch、lr 在公开预览中不完整，需人工核查。

### 7. 方法优势

高阶正样本能把多个模态/图像的共同结构放在一组；teacher-only hypergraph 适合训练期增强而不增加分类推理成本。

### 8. 局限和风险

跨患者图依赖采样；生成模块可能传播偏差；不研究模态缺失；节点粒度和完整协议仍需人工核查。

### 9. 对我的方法的启发

- **可以直接借鉴：** 训练期关系蒸馏、推理期不保留额外图。
- **需要改造后借鉴：** 把跨患者顶点改为同一 ROI 的不同可用模态/缺失视图；蒸馏 5 个节点和超边响应。
- **不建议直接借鉴：** 把不同患者放进当前 inference hypergraph，或引入 CycleGAN 式生成作为 MVP。

### 10. 阅读优先级

**方法启发；重点看 hyperedge positive definition 和 teacher-to-student distillation。**

## ViG-UNet: Vision Graph Neural Networks for Medical Image Segmentation

### 1. 基本信息

- 年份/来源：2023，ISBI，正式发表；[论文 PDF](https://april.zju.edu.cn/wp-content/papercite-data/pdf/jiang2023vig.pdf)。
- 任务/数据：ISIC 2016/2017、Kvasir-SEG。
- 代码：[声明仓库](https://github.com/juntaoJianggavin/ViG-UNet) 截至核查日为空，不应写为可复现代码。

### 2. 论文解决的问题

作者用普通 vision graph 在 U-shaped 网络各尺度建立远程 patch 关系，以补充 CNN 局部感受野。它使用 KNN 二元边，不是 hypergraph。

### 3. 方法的整体思想

一句话：**把各尺度 feature patches 作为节点，以 K=9 动态 KNN 和 max-relative graph convolution 替换/补充卷积。**

```text
image → stem → [Grapher + FFN] encoder
              → graph bottleneck
              → [Grapher + FFN] decoder + skip add
              → segmentation
```

### 4. 节点、边和超边定义

patch 是节点；每个节点连接 9 个特征近邻，边是有向二元边。图随特征层变化动态更新，无超边。

### 5. 图如何嵌入分割网络

encoder、bottleneck、decoder 都进行图卷积，skip 采用相加；图输出保留规则 patch grid，可直接上采样。

### 6. 损失函数和监督信号

`0.5 BCE + Dice`；使用旋转、翻转、归一化和 cosine LR schedule。

### 7. 方法优势

说明关系推理可放在多个尺度；max-relative aggregation 对局部差异敏感。

### 8. 局限和风险

约 0.7G 参数；非 3D/脑 MRI；动态图对缺失模态不稳定；代码仓库为空；没有医学 anatomy prior。

### 9. 对我的方法的启发

- **可以直接借鉴：** 若未来加关系模块，优先放 bottleneck/低分辨率层。
- **需要改造后借鉴：** patch graph 改为 5 节点固定 anatomy graph/hypergraph。
- **不建议直接借鉴：** 全 encoder-decoder 图化或复制其参数规模。

### 10. 阅读优先级

**方法启发；只需重点看多尺度放置和 graph convolution，代码部分快速阅读。**

## GraphCL: Graph-based Clustering for Semi-supervised Medical Image Segmentation

### 1. 基本信息

- 年份/来源：2025，ICML，正式发表；[PMLR](https://proceedings.mlr.press/v267/wang25cq.html)。
- 任务/数据：ACDC、LA、Pancreas-NIH 半监督分割。
- 代码：[GraphCL](https://github.com/dreamkily/GraphCL)，已公开。

### 2. 论文解决的问题

少标注分割中的伪标签会形成类间混叠。GraphCL 在跨样本 instance graph 上执行 GCN 对齐和无需预设簇数的聚类目标。该图不是患者内 anatomy graph。

### 3. 方法的整体思想

一句话：**把训练样本表示组成 dense ordinary graph，以图聚类约束半监督分割的表示结构。**

```text
labeled/unlabeled images → teacher-student segmentation features
                         → sample nodes + pairwise affinities
                         → GCN alignment / k-less clustering
                         → segmentation consistency
```

### 4. 节点、边和超边定义

每个样本的 CNN 表示是节点；边是由局部图像/原始特征计算的 pairwise affinity；结构跨患者、训练期动态，无超边。

### 5. 图如何嵌入分割网络

图作为 teacher-student 半监督训练的 representation regularizer，推理仍由 U-Net/V-Net 完成。

### 6. 损失函数和监督信号

监督分割、pseudo-label/consistency 和 graph clustering objectives；官方代码分别提供 ACDC、LA、Pancreas 训练脚本。

### 7. 方法优势

图仅在训练期约束表示，不增加临床推理图；能利用未标注数据。

### 8. 局限和风险

跨患者 batch/memory 依赖；普通 dense graph 可能放大错误伪标签；不处理多模态/缺失；与当前 5 个 ROI 的语义层级不同。

### 9. 对我的方法的启发

- **可以直接借鉴：** 训练期一致性正则、推理期移除辅助结构。
- **需要改造后借鉴：** 用同一患者不同缺失组合形成正对，而不是跨患者 dense graph。
- **不建议直接借鉴：** 在线推理依赖其他患者或把样本当作当前超图节点。

### 10. 阅读优先级

**方法启发；重点看训练期 graph regularization，而非网络主体。**

## D3Seg: Disentangled Diffusion and Dynamic Graph Fusion for Missing Modality Brain Tumor Segmentation

### 1. 基本信息

- 年份/来源：2026，[arXiv:2605.22249](https://arxiv.org/abs/2605.22249)，截至核查日为预印本。
- 任务/数据：BraTS 2023 Glioma 与 279 例外部 meningioma 子集；全部 15 种模态组合。
- 代码：未发现官方仓库。

### 2. 论文解决的问题

作者认为缺失模态应同时处理可用模态关系和关键缺失内容，因此使用动态模态图融合，并仅对缺失 T1ce/FLAIR 做 latent diffusion completion。多跳 adjacency 仍是 ordinary graph，不是 hypergraph。

### 3. 方法的整体思想

一句话：**以四个模态描述符构成 availability-aware 普通图，在两尺度融合可用模态关系，并对关键缺失模态进行受限扩散补全。**

```text
available MRI → modality-specific 3D encoders → 4 modality nodes
              → masked cosine A, A², A³ → two-scale MMGF
missing T1ce/FLAIR → latent diffusion completion
              → re-masked context → EGDR decoder → segmentation
```

### 4. 节点、边和超边定义

四个模态全局描述符为节点；cosine similarity 为二元边，availability mask 清零缺失模态行/列；`A²/A³` 是普通图多跳传播，不产生一条含多个节点的超边。

### 5. 图如何嵌入分割网络

MMGF 位于两个 encoder scales，输出进入 decoder；补全与关系融合协同，融合后再次 masking 以避免虚假可用性。

### 6. 损失函数和监督信号

联合 segmentation 和 diffusion/completion objectives。公开稿对 mask sampling、epoch、batch、checkpoint 规则仍不够完整，标记人工核查。

### 7. 方法优势

把 availability mask 直接写入关系矩阵；报告全 15 组合和外部肿瘤类型；限制补全目标，避免无差别生成所有模态。

### 8. 局限和风险

仅预印本、无代码；扩散计算重；动态 cosine 图会随缺失模式漂移；方法贡献多，难以判断图融合单独作用。

### 9. 对我的方法的启发

- **可以直接借鉴：** availability mask 进入 relation/hyperedge weight；固定模态槽位。
- **需要改造后借鉴：** 模态图信息作为现有 ROI 模态 gate 的输入，而不是替代 anatomy hypergraph。
- **不建议直接借鉴：** MVP 引入 diffusion 或把 `A²/A³` 宣称为高阶超图。

### 10. 阅读优先级

**方法启发；重点看 availability masking。预印本结论谨慎使用。**

## Adversarial Co-training Network for Brain Tumor Segmentation with Missing Modalities

### 1. 基本信息

- 年份/来源：2021，MICCAI，正式发表；[论文页](https://miccai2021.org/openaccess/paperlinks/2021/09/01/039-Paper0901.html)。
- 任务/数据：BraTS 2018，285 例缺失模态脑肿瘤分割。
- 代码：[ACN](https://github.com/Wangyixinxin/ACN)，已公开。

### 2. 论文解决的问题

缺失网络不能直接获得完整模态互补信息。ACN 让完整模态网络和缺失子集网络 adversarial co-train，通过 entropy/knowledge alignment 迁移知识。无显式图。

### 3. 方法的整体思想

一句话：**以完整模态网络作为知识源，通过熵和特征对抗对齐训练各缺失模态分割网络。**

```text
full modalities → complete network ─┐
missing subset → subset network ────┼→ entropy/knowledge discriminators
                                    └→ tumor segmentation
```

### 4. 节点、边和超边定义

无节点/边/超边；完整与缺失 feature maps 的 adversarial alignment 不等于图。

### 5. 分割如何嵌入网络

分割是主任务，完整和缺失分支并行；对抗模块仅在训练时使用。

### 6. 损失函数和监督信号

segmentation、entropy adversarial、knowledge adversarial objectives。覆盖多种缺失子集，但通常为不同子集训练专用模型；独立测试集和 mask sampling 细节需人工核查。

### 7. 方法优势

建立了 full-to-missing teacher-like transfer 的清晰范式；辅助模块可在推理时去除。

### 8. 局限和风险

多模型部署；强依赖完整模态病例；对抗训练不稳定；不能直接满足单模型 15 组合。

### 9. 对我的方法的启发

- **可以直接借鉴：** 用完整组合作为训练期 teacher，缺失组合作为 student。
- **需要改造后借鉴：** 只蒸馏 segmentation logits、5 个 node embeddings 或 hyperedge responses；一个共享 student 处理全部 masks。
- **不建议直接借鉴：** 为 15 组合建立 15 个 classifier，或初始阶段使用 adversarial loss。

### 10. 阅读优先级

**方法启发；重点看完整/缺失协同协议及其数据假设。**

## Scratch Each Other's Back: Incomplete Multi-Modal Brain Tumor Segmentation via Category Aware Group Self-Support Learning

### 1. 基本信息

- 年份/来源：2023，ICCV，正式发表；[论文](https://openaccess.thecvf.com/content/ICCV2023/html/Qiu_Scratch_Each_Others_Back_Incomplete_Multi-Modal_Brain_Tumor_Segmentation_via_ICCV_2023_paper.html)。
- 任务/数据：BraTS 2015/2018/2020，全 15 组合。
- 代码：[GSS](https://github.com/qysgithubopen/GSS)，已公开。

### 2. 论文解决的问题

不同模态对不同肿瘤类别的敏感性不同；统一蒸馏会把弱模态噪声传播给强模态。GSS 按类别分组，由高置信 sensitive leader 产生共同软目标。它是 group relational reasoning，不是 graph/hypergraph。

### 3. 方法的整体思想

一句话：**按肿瘤类别组织可用模态预测，以高置信 leader 和随机空间 soft-label mask 实现模态间自支持。**

```text
modality-specific predictions → category-aware groups
                              → sensitive leader / common target
                              → confidence-selected soft supervision
                              → base segmentation network
```

### 4. 节点、边和超边定义

没有 incidence matrix；group 是按类别组织的预测集合，不能称为超边。

### 5. 关系监督如何嵌入网络

作为 RFNet/mmFormer 的训练期插件；不改变推理网络。随机 soft-label mask 仅选择约 20% 高置信空间位置。

### 6. 损失函数和监督信号

base segmentation loss 加 category-aware group self-support consistency loss；伪标签由同一病例不同模态预测产生。

### 7. 方法优势

避免把所有模态等权互教；置信度筛选降低错误伪监督；训练插件与主网络解耦。

### 8. 局限和风险

leader 预测错误可能自强化；训练目标仍以分割为主；没有分类收益；group 关系缺少固定医学拓扑。

### 9. 对我的方法的启发

- **可以直接借鉴：** 置信度筛选、按 ROI/类别组织一致性。
- **需要改造后借鉴：** 同一 ROI 在不同可用模态/缺失组合下的 node embedding 互相约束；低置信 boundary 不参与蒸馏。
- **不建议直接借鉴：** 用单一强模态作为永久 teacher，或把 group 命名为 hyperedge。

### 10. 阅读优先级

**方法启发；重点看 category-aware confidence selection。**

## Enhancing Modality-Agnostic Representations via Meta-Learning for Brain Tumor Segmentation

### 1. 基本信息

- 年份/来源：2023，ICCV，正式发表；[论文 PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Konwer_Enhancing_Modality-Agnostic_Representations_via_Meta-Learning_for_Brain_Tumor_Segmentation_ICCV_2023_paper.pdf)。
- 任务/数据：BraTS 2018/2019/2020 及两个异构临床 cohort。
- 代码：未发现官方仓库。

### 2. 论文解决的问题

许多缺失模态方法默认所有训练病例完整，再人为 dropout；现实中完整病例也可能只占少数。该方法以 partial-modality tasks 做 meta-train，以有限完整病例做 meta-test，并用 discriminator 对齐完整/缺失表示。

### 3. 方法的整体思想

一句话：**通过 meta-learning 让共享参数模态 encoders 从有限完整病例快速适应多种部分模态任务，并对抗性学习 modality-agnostic representation。**

```text
partial-modality tasks → meta-train shared encoders
limited full cases     → meta-test / outer update
full vs missing latent → modality discriminator
                      → SwinUNETR decoder → segmentation
```

### 4. 节点、边和超边定义

无图。固定模态 slots、meta tasks 和 discriminator 都不构成图。

### 5. 分割如何嵌入网络

分割是主任务，meta-learning 决定训练组织；adversarial discriminator 只约束 latent representations。

### 6. 损失函数和监督信号

segmentation、meta inner/outer objectives、modality-invariance adversarial loss。主设置仅 50% 受试者有完整模态；128³，per-task batch 1、meta-batch 8，3×RTX8000。

### 7. 方法优势

明确打破“训练阶段必有完整模态”的隐藏假设；扩展到异构临床 cohort；适合评估天然缺失数据。

### 8. 局限和风险

训练复杂、算力高；fixed slots/zero fill 仍可能产生模态捷径；无图和分类任务；代码不可用降低复现性。

### 9. 对我的方法的启发

- **可以直接借鉴：** 分层报告完整病例训练与天然不完整训练；按 availability pattern 组织任务。
- **需要改造后借鉴：** 根据缺失严重度调节 segmentation auxiliary weight，或在有限完整病例上建立 teacher consistency。
- **不建议直接借鉴：** 在尚未验证简单 auxiliary head 前引入二阶/复杂 meta-learning。

### 10. 阅读优先级

**方法启发；重点看数据可用性假设和 50% 完整病例 protocol。**

### C. 背景参考

## Graph Flow: Cross-Layer Graph Flow Distillation for Dual Efficient Medical Image Segmentation

### 1. 基本信息

- 年份/来源：2023，IEEE TMI，正式发表；[PubMed](https://pubmed.ncbi.nlm.nih.gov/36423314/)。
- 任务/数据：胃癌、Synapse、BUSI、CVC-ClinicDB 分割。
- 代码：未核实到可用官方仓库。

### 2. 论文解决的问题

普通 logits distillation 只对齐最终输出，无法传递 encoder 跨层关系变化。Graph Flow 以通道 salience graph 蒸馏 teacher/student 的节点与边变化。它是 relation distillation，不是 anatomy graph。

### 3. 方法的整体思想

一句话：**把特征通道作为节点、显著区域相关性作为边，蒸馏相邻层之间的 graph variation。**

```text
teacher/student features → channel salience patches
                         → channel-correlation graphs
                         → cross-layer graph variation alignment
                         → efficient student segmentation
```

### 4. 节点、边和超边定义

节点是通道；每个节点取最大激活附近 patch；边是两通道 salience regions 的相关性。均为二元边，无超边。

### 5. 图如何嵌入分割网络

图仅计算蒸馏损失，不替代 encoder/decoder；推理使用 Mobile U-Net、ENet 或 ERFNet student。

### 6. 损失函数和监督信号

segmentation loss、graph-flow distillation、logits/adversarial distillation；Adam、`wd=2e-4`，部分数据使用 3×3/9×9 salience patch。

### 7. 方法优势

提供一种“不复制 teacher 架构、只蒸馏关系变化”的轻量路径。

### 8. 局限和风险

通道节点缺乏 anatomy 语义；不研究多模态/缺失；教师偏差会传给学生；无可核实代码。

### 9. 对我的方法的启发

- **可以直接借鉴：** 蒸馏 full/missing 情况下同层到跨层的关系变化。
- **需要改造后借鉴：** 通道 graph 改为 5 个 node/hyperedge responses。
- **不建议直接借鉴：** 同时加入 adversarial、logit 和 graph 三类蒸馏，造成过度约束。

### 10. 阅读优先级

**仅作为关系蒸馏背景；重点看 variation graph 定义。**

## Development and validation of a deep learning model for breast lesion segmentation and characterization in multiparametric MRI

### 1. 基本信息

- 年份/来源：2022，Frontiers in Oncology，正式发表；[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9402900/)。
- 任务：DCE/DWI 病灶分割后良恶性分类；含内部与外部 cohort。
- 代码：未发现官方仓库。

### 2. 论文解决的问题

全乳输入背景大，病灶分割可生成分类 ROI。该工作采用分阶段 cascade，不是共享 encoder 多任务，也没有 graph。

### 3. 方法的整体思想

一句话：**分别在 DCE 和 DWI 上分割病灶，再将预测 mask 引导的 3D 体积输入 ResNet 分类器。**

```text
DCE → 2D V-Net ─┐
DWI → Attention U-Net ─┼→ predicted lesion masks → masked 3D volumes
                       → 3D ResNet → benign/malignant
```

### 4. 节点、边和超边定义

无图；预测 mask 只是 ROI，不是节点/边结构。

### 5. 分割如何嵌入分类网络

分割位于分类前处理阶段，mask 直接裁剪/关注分类输入；两阶段分别训练。

### 6. 损失函数和监督信号

segmentation：Adam、80 epochs、batch 16、`lr=1e-4`；classification：batch 12，后续 `lr=1e-3`；含翻转和 90°旋转。

### 7. 方法优势

大样本和外部验证支持 segmentation-guided ROI 的实际可用性；分阶段训练避免梯度冲突。

### 8. 局限和风险

hard mask 会把漏分病灶直接删掉；非脑 MRI；不处理缺失模态；不能证明分割监督改善共享 encoder。

### 9. 对我的方法的启发

- **可以直接借鉴：** 把 predicted mask 当作辅助空间先验，而不是标签本身。
- **需要改造后借鉴：** hard crop 改为 soft weighted pooling，并与现有 ROI masks 残差混合。
- **不建议直接借鉴：** 分割失败时完全屏蔽节点，或使用真实测试 mask。

### 10. 阅读优先级

**背景参考；用于理解 segmentation-to-classification cascade 的收益与错误传播。**

---

## 9.1 超图在医学影像分割中通常建模什么

| 节点层级 | 典型关系/超边 | 语义性 | 复杂度 | 标签依赖 | 缺失模态敏感性 | 对当前 5 节点适配性 |
|---|---|---:|---:|---:|---:|---:|
| voxel-level | 相似体素、前景/背景 prototype | 中 | 极高 | 可不依赖，但常受 coarse mask 引导 | 高 | 低 |
| patch-level | KNN 相似 patch、局部窗口 | 中 | 高 | 低 | 高，特征距离会漂移 | 低到中 |
| superpixel/supervoxel-level | 空间邻接或相似区域 | 中高 | 中高 | 依赖预处理，不一定依赖 GT | 中 | 中 |
| anatomical-region-level | 解剖邻接、病灶-边界-周边联合关系 | 高 | 低 | 若用预测 mask 可避免测试 GT | 中低 | **最高** |
| tumor-subregion-level | enhancing/core/edema 等共同超边 | 高 | 低 | 训练依赖亚区标签；推理用预测 posterior | 中 | 高 |
| modality-level | 可用模态相似性、互补关系 | 中 | 极低 | 低 | 直接受 availability 影响 | 作为融合辅助，高 |
| patient-level | 跨病例相似性/聚类超边 | 低到中 | 中 | 低 | 中 | 在线推理不推荐 |
| multi-scale hypergraph | 多分辨率节点和跨尺度关系 | 中高 | 高 | 低到中 | 高 | 当前阶段不推荐 |

**综合判断。** AHGNN/ReHyDIL 的 patch 或 patient-spatial hypergraph 证明了高阶传播的可行性，但会重新引入 sample-wise KNN 和批次依赖。UHR-Net 的 soft incidence 最适合抽象为“节点属于医学关系的可信程度”。当前模型已经有语义更强、计算更低的 anatomy-region nodes，因此应保留节点身份和固定 incidence matrix，把学习自由度放在 **soft ROI membership、node confidence、hyperedge residual weights** 上。

## 9.2 超图位于分割网络的哪个位置

| 放置位置 | 作用 | 优点 | 风险 | 对分类主任务的建议 |
|---|---|---|---|---|
| encoder-level | 早期建立关系 | 可影响全部下游特征 | 高分辨率成本大；容易受模态噪声影响 | 不建议作为 MVP |
| bottleneck hypergraph | 深层语义关系 | 成本低、与分类 embedding 接近 | 细边界信息较弱 | 适合未来节点/关系一致性 |
| decoder-level | posterior/边界细化 | 直接优化 mask | 对分类 encoder 反向影响可能弱 | 轻量辅助头可采用 |
| skip-connection reasoning | 对齐细节与语义 | 保留边界 | 多尺度实现复杂 | 暂不采用 |
| multi-scale graph/hypergraph | 同时建模大小结构 | 表达充分 | 参数和归因成本高 | 不适合作为第一阶段 |
| segmentation posterior refinement | 用 coarse mask 和不确定性再推理 | 可直接产生 soft ROI/confidence | 错误闭环 | 适合方案 B/C，但需 residual fallback |

对于“分类主任务、分割辅助任务”，最佳初始位置是：**共享各模态 encoder 的最后一到两层特征，接一个独立轻量 segmentation decoder；不让辅助 decoder 改写现有 HGNN 输入。** 这样最容易回答“dense spatial supervision 是否改善分类 encoder”。只有在这一问题为真后，才把 posterior 接入 node construction。

## 9.3 分割辅助分类的主要范式

### 模式一：共享编码器 + 双任务头

```text
Shared modality encoders
   ├── existing ROI pooling → anatomy HGNN → classification
   └── lightweight decoder → segmentation
```

- 优点：实现成本最低；训练期 dense supervision 可直接改善 encoder；推理可移除分割头。
- 风险：分割梯度更密集，可能压制分类；共享得太浅则收益有限，共享得太深则冲突增大。
- 缺失模态适配：强。对分割和分类使用同一 availability mask，并只监督可用模态分支。
- 当前适配性：**最高，是 MVP。**

### 模式二：分割结果引导分类

```text
segmentation posterior → soft ROI masks → ROI pooling → anatomy HGNN → classification
```

- 优点：直接改善 core/boundary/peritumoral node construction；可解释性强。
- 风险：形成 error propagation；hard mask 尤其危险；推理必须运行分割头。
- 缺失模态适配：中等，必须使用 entropy/confidence 和 baseline ROI fallback。
- 当前适配性：高，但应在模式一验证后实施。

### 模式三：分割特征与分类特征交互

```text
segmentation decoder features ↔ ROI/node features → HGNN → classification
```

- 优点：不必把 mask 二值化；亚区和边界语义可进入 nodes。
- 风险：接口复杂，可能让分类器依赖 decoder-specific shortcuts；推理必须保留部分 decoder。
- 当前适配性：中等，适合作为后续消融中的“segmentation feature only”。

### 模式四：分割用于预训练或正则化

```text
segmentation pretraining → initialize encoders → classification fine-tuning
```

- 优点：没有联合梯度冲突；推理零开销；标签泄漏风险容易控制。
- 风险：fine-tuning 可能遗忘空间表征；不能利用样本级 segmentation uncertainty。
- 当前适配性：中高，可作为独立 baseline 判断联合训练是否必要。

### 模式五：完整模态教师—缺失模态学生

```text
full-modality teacher segmentation/nodes
                 ↓ distillation
missing-modality student → classification
```

- 优点：完整模态 segmentation posterior 为缺失 student 提供结构目标；适合 node/hyperedge consistency。
- 风险：训练必须有足够完整病例；teacher 错误被固化；多重 loss 增加冲突。
- 当前适配性：中等。只有在数据确实具有完整训练病例时采用；天然不完整数据应参考 meta-learning 的 protocol。

## 9.4 分割监督如何作用于 anatomy hypergraph

### 9.4.1 从 posterior 生成 soft anatomy masks

设分割头输出肿瘤/亚区概率 (p(x))，brain mask 为 (B(x))。可以构造：

\[
M_{core}=p_{core},\qquad
M_{boundary}=\operatorname{SoftMorphGrad}(p_{tumor}),
\]

\[
M_{peri\_inner}=\operatorname{SoftDilate}_{r_1}(p_{tumor})-p_{tumor},
\]

\[
M_{peri\_outer}=\operatorname{SoftDilate}_{r_2}(p_{tumor})-
\operatorname{SoftDilate}_{r_1}(p_{tumor}),
\]

\[
M_{distal}=B-\operatorname{SoftDilate}_{r_2}(p_{tumor}).
\]

这些 mask 必须裁剪到 ([0,1]) 并归一化。**测试阶段只能使用预测 posterior，不能使用真实 mask。** 为避免误差直接污染节点，先与现有 ROI masks (M_r^{base}) 残差混合：

\[
\widetilde M_r=(1-\alpha)M_r^{base}+alpha\,\operatorname{stopgrad}(M_r^{seg}),
\quad \alpha:0\rightarrow 0.3.
\]

### 9.4.2 soft pooling 与不确定性

weighted mean 可写为：

\[
\mu_r=\frac{\sum_x \widetilde M_r(x)F(x)}{\sum_x\widetilde M_r(x)+\epsilon}.
\]

hard max 对单个错误 voxel 很敏感，建议用 temperature-controlled log-sum-exp 或 softmax pooling 替代。分割熵

\[
u(x)=-\sum_k p_k(x)\log p_k(x)
\]

可聚合为 node confidence (c_r=1-\mathbb{E}_{x\sim M_r}[u(x)]/\log K)。当 (c_r) 低或 soft mask 体积异常时，回退到 (M_r^{base})，而不是删除节点。

### 9.4.3 分割特征与超边权重

segmentation decoder feature 可以先 ROI pooling，再与现有 modality-fused node feature 拼接或以小门控注入。为控制 shortcut，初始实验应对 segmentation feature 使用 stop-gradient。对于包含 boundary 的超边 E1/E2/E5，可引入 boundary confidence；E3/E4 更侧重 peri/distal volume 与不确定性。缺失模态时，把 availability mask embedding 与 node uncertainty 一起输入边权网络。

## 9.5 多任务损失与负迁移

总目标为：

\[
\mathcal L=\mathcal L_{cls}+\lambda_{seg}\mathcal L_{seg}+
\lambda_{aux}\mathcal L_{aux}.
\]

### 权重与优化策略比较

- **固定权重：** 最容易解释；建议先从 `λ_seg=0.1` 开始，验证后再搜索 `{0.05, 0.1, 0.2}`。
- **uncertainty weighting：** 可学习任务噪声，但可能通过增大方差项自动弱化困难任务；参见 [Kendall et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)。
- **GradNorm：** 平衡任务梯度范数；适合长期训练，但改变主任务优先级；参见 [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)。
- **PCGrad：** 仅在任务梯度内积为负时投影冲突分量；适合确认冲突后使用；参见 [PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)。
- **DWA/attention MTL：** 根据损失下降速度动态调整，适合作为后续比较而非默认；参见 [MTAN/DWA](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html)。
- **curriculum/warm-up：** 先稳定 segmentation head，再以小权重联合；或先 classification baseline，再逐步增加 `λ_seg`。
- **stop-gradient/partial sharing：** 对 soft ROI、confidence gate 或浅层 modality branch 截断梯度，是最直接的冲突隔离手段。

### 分割何时损害分类

1. segmentation 强调边界/体素一致性，而分类依赖全局异质性、分子表型或远端 context；
2. decoder 梯度规模远大于小型 classifier，encoder 被优化成“可分割”而非“可分类”；
3. 强模态提供更准确 mask，使 encoder/gate 过度依赖 T1ce/FLAIR，反而降低严重缺失组合稳定性；
4. hard ROI 形成错误闭环，分类无法从被 mask 掉的区域恢复；
5. 同一病例使用由真实测试 mask 构建的 nodes，造成标签泄漏和不可部署评估。

### 如何检测与控制

- 每个 epoch 记录共享 encoder 各层的 `cos(g_cls, g_seg)` 和梯度范数比；连续负余弦才说明真实冲突。
- checkpoint 只按 validation classification `balanced_accuracy` 选择；segmentation Dice 是诊断指标，不是选择标准。
- 分层报告 15 个组合、worst-combination 和按可用模态数分组的结果。
- segmentation 梯度只进入**可用模态** encoders；对缺失 slot 不产生伪梯度。
- 严重缺失组合可按 segmentation confidence 缩小有效 `λ_seg`，但第一阶段先用固定权重，避免引入新的学习捷径。
- 若冲突集中在浅层，改为部分共享；若只在 posterior→ROI 闭环出现，保留 stop-gradient；只有实测冲突后再引入 PCGrad。

### Foundational Work（不占 2021—2026 主体）

Kendall uncertainty weighting、GradNorm、DWA/MTAN 和 PCGrad 分别提供噪声建模、梯度范数平衡、损失速率平衡和冲突梯度投影。它们不是医学超图论文，但用于解释“怎样防止辅助分割压制分类”。当前研究不应同时使用多种自适应权重方法；应先建立固定权重基线。

---

## 10. 针对当前 5 节点 anatomy hypergraph 的三个候选方案

### 方案 A：共享编码器 + 轻量辅助分割头

**结构。** 保留四个独立 modality encoders、ROI mean/max pooling、ROI-wise modality attention、固定 incidence matrix、HGNN 和 node-attention pooling。只在每个 modality encoder 的最后一层语义 feature map 后接共享参数的轻量 segmentation head：`1×1×1 projection → two lightweight upsample-conv blocks → K-class logits`。可用模态 logits 通过 availability-aware normalized mean 或现有 gate 的 detached 权重融合。

**共享范围。** 共享每模态 backbone 全部已有 feature extractor；segmentation decoder 独立。若梯度冲突集中在前两层，可仅让后两层接收 segmentation gradient。

**损失。** `L_seg = Dice + CE + 0.1~0.2 BoundaryLoss`；初始 `λ_seg=0.1`，搜索 `{0.05,0.1,0.2}`。分类损失保持原定义。前 2–3 epochs 冻结 backbone 只训练 segmentation head，再联合微调。

**推理。** MVP 中完全丢弃 segmentation head；分类路径与当前模型相同，因此不存在测试时真实 mask 依赖，额外推理开销为零。

**优点与风险。** 实现成本最低，能隔离回答“分割 dense supervision 是否改善 encoder”。不足是尚未直接改善 ROI node construction；这是刻意保留的可归因设计。

### 方案 B：分割引导的 soft anatomy nodes

**结构。** 在方案 A 上保留 fused segmentation posterior，按 9.4 的 soft morphology 构建 `core/boundary/peri_inner/peri_outer/distal_normal` masks。现有 node feature 的 mean/max pooling 改为 soft weighted mean + log-sum-exp pooling。

**误差控制。** 使用 `M=(1-α)M_base+α stopgrad(M_seg)`，`α` 线性 warm-up 至 0.2–0.3；segmentation entropy 高、ROI 体积超出训练分位区间或 mask 为空时令 `α→0`。不能用 hard argmax mask 直接替换 baseline ROI。

**缺失模态稳定性。** segmentation head 与分类使用相同 modality mask；将各模态 posterior 的 disagreement 计入 uncertainty；可在 full/missing view 之间蒸馏 soft masks，但只在训练数据确有完整模态时使用 full teacher。

**优点与风险。** 它最直接改善 anatomy node 语义一致性；风险是形成 segmentation→ROI→classification 的误差闭环，因此必须先 stop-gradient、保留 fallback，并单独比较“segmentation feature only vs mask guidance”。

### 方案 C：分割置信度驱动的固定超边权重

**结构。** 不改变当前五条超边成员关系。对每条超边 (e) 计算：

\[
w_e=1+\rho\left(2\sigma\left(\operatorname{MLP}[z_e,c_e,m]\right)-1\right),
\quad \rho\in[0.2,0.3].
\]

- (z_e)：该超边成员节点 embedding 的 mean/max summary；
- (c_e)：成员 ROI 的 `1-entropy`、boundary confidence、soft volume 与 volume plausibility；
- (m)：4-bit availability mask 的 embedding；
- (w_e)：乘在现有 hypergraph normalization/message 上的样本级 residual weight。

**为什么只调权重。** 固定 E1–E5 表示医学先验，缺失模态改变的是证据可靠性，不应改变 core-boundary-peritumoral 关系是否存在。residual gate 把权重限制在 `[1-ρ,1+ρ]`，比重新 KNN 建拓扑稳健且可解释。

**防错机制。** 初期对 (c_e) stop-gradient；对 gate 加 `||w-1||²` 正则；低 segmentation confidence 时强制 (w_e→1)；不允许一条边被置零。重点监控 E1/E2/E5 的 boundary gate 是否在缺少 T1ce/FLAIR 时系统性塌缩。

**必要消融。** fixed vs learned weights；不含 segmentation confidence；不含 modality mask；无 residual bound；stop-gradient vs end-to-end；按超边报告平均权重和 15 组合变化。

---

## 11. 推荐最小可行方案（MVP）

### 11.1 网络结构与数据流

选择 **方案 A**。不修改当前 5 节点、5 超边、ROI pooling、modality gate、HGNN 和 classifier。

```text
available T1/T1ce/T2/FLAIR
        ↓ modality-specific existing encoders
        ├→ existing ROI mean/max pooling
        │   → ROI-wise mask-aware modality fusion
        │   → fixed 5-node / 5-hyperedge HGNN
        │   → node attention pooling → classification
        └→ lightweight shared segmentation head (training only)
            → tumor/subregion logits → auxiliary loss
```

### 11.2 损失函数

\[
\mathcal L=\mathcal L_{cls}
+\lambda_{seg}\left(\mathcal L_{Dice}+\mathcal L_{CE}
+\beta\mathcal L_{boundary}\right),
\]

初始 `λ_seg=0.1, β=0.1`；仅当 segmentation label 存在时计算；按可用模态数归一化，避免四模态样本获得四倍辅助权重。暂不使用 adaptive weighting、teacher、soft ROI 或 hyperedge gate。

### 11.3 训练阶段

1. 从当前 classification-only 最佳 checkpoint 初始化，冻结分类 protocol 和数据 split。
2. 训练 segmentation head 2–3 epochs，backbones/HGNN/classifier 冻结，确认 loss 可下降。
3. 联合 fine-tune；segmentation head 用基准学习率，backbones 用 `0.1×` 学习率，HGNN/classifier 沿用当前设置。
4. 每个样本采样非空 modality mask；分类和分割分支使用同一个 mask；训练分布覆盖 15 组合或当前既定 targeted schedule。
5. checkpoint 只按 validation balanced accuracy；至少 3 seeds，同时记录 segmentation Dice、gradient cosine 和 per-combination classification。

### 11.4 推理阶段

丢弃 segmentation head；继续使用当前 ROI masks 与固定 anatomy hypergraph。这样第一阶段结果只能归因于“训练期 segmentation representation regularization”，不会被 soft ROI 或额外推理混杂。

### 11.5 建议超参数

| 参数 | 初始值 | 搜索范围/说明 |
|---|---:|---|
| `λ_seg` | 0.1 | `{0.05, 0.1, 0.2}`；先固定 |
| boundary loss `β` | 0.1 | `{0, 0.1, 0.2}` |
| head warm-up | 3 epochs | 2–5 |
| backbone LR multiplier | 0.1 | 0.05–0.2 |
| segmentation head channels | 16–32 | 不超过当前 encoder output 规模 |
| gradient clipping | 沿用当前设置 | 保持实验可比 |
| seeds | 3 | 最低要求 |

### 11.6 最小实现步骤

1. 暴露四个 modality encoder 的空间 feature maps，同时保持现有 node feature API 不变。
2. 添加共享轻量 segmentation head 和 availability-aware logits fusion。
3. dataset 返回训练 segmentation label；确认 augmentation 对 MRI 和 mask 同步。
4. trainer 增加可开关的 `L_seg`，默认 `λ_seg=0` 可严格复现 baseline。
5. 日志记录 `L_cls/L_seg`、各自梯度范数、gradient cosine、segmentation Dice、15 组合分类指标。
6. 先运行 classification-only 回归测试，再运行单 seed smoke test，最后 3-seed 主比较。

### 11.7 可能失败的原因

- 分类标签与分割标签样本并不完全重合，导致有效训练集变化；
- segmentation labels 的定义与现有 `core/boundary/peri` ROI 生成规则不一致；
- decoder 梯度压制分类，需要降低 `λ_seg` 或 partial sharing；
- modality dropout 下分割过难，模型学到 availability shortcut；
- encoder feature resolution太低，辅助头只有粗 mask，无法真正改善 boundary perception；
- classification gain 只来自额外训练步数或参数，应以等训练预算 control 排除。

---

## 12. 必须设计的消融实验

| # | 对照 | 回答的科学问题 | 关键控制 |
|---:|---|---|---|
| 1 | Classification only | 当前模型基线和随机波动多大？ | 相同 split、steps、seeds |
| 2 | Classification + auxiliary segmentation | dense spatial supervision 是否提升分类和缺失稳健性？ | 只新增 head/loss；推理丢弃 head |
| 3 | Hard ROI vs soft segmentation-guided ROI | soft membership 是否减少分割错误传播？ | 同一 posterior、相同 pooling 容量 |
| 4 | Segmentation feature only vs segmentation mask guidance | 收益来自语义 feature 还是空间 mask？ | 参数量匹配；分别 stop-gradient |
| 5 | Fixed anatomy hypergraph vs segmentation-weighted hypergraph | segmentation confidence 是否应改变关系传播强度？ | 拓扑始终固定，只变 edge weights |
| 6 | Shared encoder vs partially shared encoder | 梯度冲突发生在哪个深度？ | 相同 head 和 loss |
| 7 | Fixed loss weight vs adaptive task weighting | 自适应平衡是否优于可解释固定权重？ | 先完成固定权重 sweep，再选一种 adaptive 方法 |
| 8 | Full-modality training vs missing-modality augmentation | robustness 来自辅助分割还是训练 mask 分布？ | 相同总样本/steps，报告 15 组合 |
| 9 | With vs without stop-gradient | posterior/feature 反向闭环是帮助还是负迁移？ | 分别在 soft ROI 和 confidence gate 测试 |
| 10 | Ground-truth ROI upper bound | ROI node construction 尚有多少理论 headroom？ | **仅分析；不得作为正式测试输入或主结果** |

还应增加三个报告维度：按可用模态数分层、worst-combination 指标、每个 hyperedge 的权重/attention 稳定性。若仅平均指标上升但 worst-combination 下降，不能声称增强了缺失模态鲁棒性。

---

## 13. 最终结论：十个明确回答

1. **近五年最具代表性的超图医学分割工作有哪些？** AHGNN、HyCon、ReHyDIL，以及截至核查日仍为预印本的 UHR-Net。脑肿瘤普通图/缺失模态近邻参考是 GNN-Tumor-Seg、AdaMM、D3Seg；不可把后三者写成超图。
2. **它们主要使用什么节点和超边？** 节点多为低分辨率像素/patch、跨患者空间位置或跨模态特征；超边由动态 KNN、局部窗口、跨模态同源组或前景/背景 prototype 的 soft incidence 构成。
3. **超图的主要价值是什么？** 用一次传播表达多节点共同关系，在非局部语义、跨模态一致性、前景/背景归属和拓扑保持上比二元边更自然；价值依赖合理节点语义，而非“高阶”名称本身。
4. **最值得精读哪些论文？** 优先顺序：ReHyDIL → AHGNN → UHR-Net → GNN-Tumor-Seg → mmFormer → Multi-task IDH/1p19q → GliomaNet → AdaMM → M3AE。
5. **哪些设计可直接迁移？** 轻量辅助 segmentation head、相同 modality dropout、低分辨率监督、soft posterior、entropy/confidence、full/missing view consistency、训练期关系蒸馏。
6. **哪些必须改造？** patch/supervoxel 节点需改为 5 个 anatomy ROI；动态 KNN 改为固定 topology + dynamic residual weights；分割主任务改为分类主、分割辅；完整 teacher 仅在数据条件允许时使用。
7. **哪些不建议采用？** voxel-level 大图、跨患者在线图、无约束 sample-wise KNN、14 adapters、diffusion completion、hard mask 直接替换 ROI、测试时真实分割标签。
8. **分割辅助最可能帮助哪个模块？** 第一是各 modality encoder 对肿瘤/边界的空间表征；第二是 ROI node construction；第三才是超边可靠性。它不太可能直接改善固定拓扑本身。
9. **最可能的负迁移是什么？** dense segmentation gradients 压过 classification gradients，使 encoder 偏向边界而丢失分型相关全局/异质性；严重缺失组合下，这种冲突可能更强。
10. **当前最值得先实现什么？** 方案 A：共享现有 modality encoders，增加训练期轻量 segmentation head，`λ_seg≈0.1`，分类路径与推理完全不变；以 validation balanced accuracy、3 seeds 和 15 组合分层结果判断是否进入 soft anatomy nodes。

---

## 证据边界与人工核查清单

- HyCon 的节点空间粒度和完整训练超参需查期刊全文/代码配置。
- UHR-Net、D3Seg 是 2026 预印本，不能按正式同行评审证据表述。
- GliomaNet 的精确 cohort、split、损失权重和两阶段/联合优化细节需人工核查。
- Meta-Radiology 2025 多任务论文的完整损失公式、权重和优化器需全文核查。
- ACN 的独立测试划分和缺失 mask 采样比例需人工核查。
- AdaMM 仓库公开但当前快照不完整，不能把“有仓库”等同于“完整可复现”。
- 本综述是面向方法设计的叙述性检索，不是 PRISMA 系统综述，也不进行跨论文数值 meta-analysis。

## AI 辅助声明

本文件使用 AI 辅助完成检索组织、证据矩阵整理和方法综合；论文事实应在正式引用或实现前由研究者对照原文 PDF、补充材料和代码再次核查。
