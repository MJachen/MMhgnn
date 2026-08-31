# Inspiration Reading List

Scope: recent 2021-2026 papers for method inspiration, not a full literature review. I prioritize papers that can inspire a missing-modality multimodal brain MRI classification model with ROI-level graph/hypergraph reasoning. Some entries are preprints; verify final venues before citation.

## Hypergraph Tversky-Aware Domain Incremental Learning for Brain Tumor Segmentation with Missing Modalities

- Year / venue: 2025 / arXiv. Source: https://arxiv.org/abs/2505.16809
- Task: Missing-modality brain tumor segmentation.
- Classification or segmentation: Segmentation.
- Graph / hypergraph / missing-modality / fusion: Hypergraph + missing-modality + domain-incremental learning.
- Why I should read it: It is the closest paper to your research space because it explicitly combines hypergraphs, missing MRI modalities, and brain tumor segmentation.
- Key idea: ReHyDIL uses a Cross-Patient Hypergraph Segmentation Network to capture high-order patient associations and a Tversky-aware contrastive loss to handle modality imbalance.
- Possible inspiration for my method: Use hyperedges to encode high-order relations, but adapt them to within-case ROI/modal relations rather than cross-patient segmentation.
- What is not applicable to my work: It is segmentation and domain-incremental learning, not HGG/LGG classification; its cross-patient hypergraph is not the same as an anatomical ROI hypergraph.
- Priority: Must read

## D3Seg: Dependency-Aware Diffusion for Brain Tumor Segmentation with Missing Modalities

- Year / venue: 2026 / arXiv. Source: https://arxiv.org/abs/2605.22249
- Task: Missing-modality brain tumor segmentation.
- Classification or segmentation: Segmentation.
- Graph / hypergraph / missing-modality / fusion: Modality graph fusion + latent diffusion imputation + missing-modality robustness.
- Why I should read it: It combines graph-style modality dependency modeling with a lightweight imputation mechanism, which is directly relevant to your possible T1ce feature-completion direction.
- Key idea: Multi-hop Modality Graph Fusion models higher-order modality dependencies, while diffusion-based latent imputation compensates missing T1ce representations.
- Possible inspiration for my method: Add a small latent completion head after ROI encoding, and test whether predicted T1ce ROI features improve no-T1ce settings without harming full-modality performance.
- What is not applicable to my work: It is still a segmentation method and focuses on latent imputation for dense prediction, not classification calibration.
- Priority: Must read

## A Joint Graph and Image Convolution Network for Automatic Brain Tumor Segmentation

- Year / venue: 2021 / arXiv, BraTS 2021 challenge method. Source: https://arxiv.org/abs/2109.05580
- Task: Brain tumor segmentation.
- Classification or segmentation: Segmentation.
- Graph / hypergraph / missing-modality / fusion: Graph learning + CNN refinement.
- Why I should read it: It is a simple example of representing each brain as a graph of image regions before using CNN refinement.
- Key idea: A graph neural network first segments graph-defined brain regions, then a voxel CNN refines the tumor prediction.
- Possible inspiration for my method: Use graph nodes as interpretable anatomical/ROI units before classification, rather than treating the entire 3D image as one opaque feature tensor.
- What is not applicable to my work: It does not handle missing modalities and does not use hypergraphs.
- Priority: Useful

## No Modality Left Behind: Adapting to Missing Modalities via Knowledge Distillation for Brain Tumor Segmentation

- Year / venue: 2025 / arXiv. Source: https://arxiv.org/abs/2509.15017
- Task: Missing-modality brain tumor segmentation.
- Classification or segmentation: Segmentation, with an auxiliary lesion-presence classification signal.
- Graph / hypergraph / missing-modality / fusion: Graph-guided adaptive refinement + teacher-student distillation + reliability modeling.
- Why I should read it: It is a recent, broad missing-modality strategy paper and explicitly compares categories of missing-modality methods.
- Key idea: AdaMM combines graph-guided adaptive refinement, bi-bottleneck teacher-student distillation, and lesion-presence-guided reliability.
- Possible inspiration for my method: Use a teacher model trained on full modalities to regularize ROI embeddings from missing-modality inputs; add an auxiliary reliability or lesion-presence proxy if it improves calibration.
- What is not applicable to my work: Dense segmentation distillation may be too heavy for a compact classification model.
- Priority: Must read

## mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation

- Year / venue: 2022 / arXiv. Source: https://arxiv.org/abs/2206.02425
- Task: Incomplete multimodal brain tumor segmentation.
- Classification or segmentation: Segmentation.
- Graph / hypergraph / missing-modality / fusion: Missing-modality + intra-modal and inter-modal Transformer fusion.
- Why I should read it: It is a core missing-modality segmentation baseline and shows how to process arbitrary modality subsets with modality-specific encoders.
- Key idea: Hybrid CNN/Transformer encoders capture within-modality context, while an inter-modal Transformer aligns global cross-modal semantics.
- Possible inspiration for my method: Separate modality-specific feature extraction from cross-modality interaction; replace global token attention with ROI-level modality hyperedges.
- What is not applicable to my work: Transformer-based dense prediction is larger and less anatomically explicit than your ROI hypergraph classifier.
- Priority: Must read

## M3AE: Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities

- Year / venue: 2023 / arXiv. Source: https://arxiv.org/abs/2303.05302
- Task: Missing-modality brain tumor segmentation.
- Classification or segmentation: Segmentation.
- Graph / hypergraph / missing-modality / fusion: Modality dropout + masked autoencoding + self-distillation.
- Why I should read it: It is one of the clearest papers for thinking about modality dropout and robust pretraining.
- Key idea: Random modalities and random patches are masked during pretraining; model inversion and self-distillation help handle heterogeneous missing-modality cases.
- Possible inspiration for my method: Pretrain ROI encoders with modality dropout or masked ROI-feature reconstruction before supervised HGG/LGG classification.
- What is not applicable to my work: Image reconstruction/substitution may not be necessary if your model is designed to reason only over observed modalities.
- Priority: Must read

## ACN: Adversarial Co-training Network for Brain Tumor Segmentation with Missing Modalities

- Year / venue: 2021 / arXiv. Source: https://arxiv.org/abs/2106.14591
- Task: Brain tumor segmentation under missing modalities.
- Classification or segmentation: Segmentation.
- Graph / hypergraph / missing-modality / fusion: Missing-modality + adversarial co-training + latent alignment.
- Why I should read it: It is a useful early template for treating missing modalities as a domain gap between full and incomplete inputs.
- Key idea: Full-modality and missing-modality models co-train each other, while adversarial losses align domains and improve prediction reliability.
- Possible inspiration for my method: Align full-modality and missing-modality ROI embeddings using a lightweight contrastive or adversarial regularizer.
- What is not applicable to my work: Training separate models for each missing situation may be impractical if your goal is one unified classifier.
- Priority: Must read

## DIGEST: Deeply Supervised Knowledge Transfer Network Learning for Brain Tumor Segmentation with Incomplete Multi-modal MRI Scans

- Year / venue: 2022 / arXiv. Source: https://arxiv.org/abs/2211.07993
- Task: Brain tumor segmentation with incomplete multimodal MRI.
- Classification or segmentation: Segmentation.
- Graph / hypergraph / missing-modality / fusion: Missing-modality + teacher-student learning.
- Why I should read it: It is a clean teacher-student formulation for transferring complete-modality knowledge to incomplete-modality students.
- Key idea: A student trained under Bernoulli-sampled missing modalities learns semantic information from a complete-modality teacher at multiple decoder stages.
- Possible inspiration for my method: Distill full-modality classifier logits and ROI embeddings into the same classifier under randomly sampled modality subsets.
- What is not applicable to my work: Decoder-stage dense supervision does not directly transfer to a patient-level classifier.
- Priority: Must read

## Multi-modal Brain Tumor Segmentation via Missing Modality Synthesis and Modality-level Attention Fusion

- Year / venue: 2022 / arXiv. Source: https://arxiv.org/abs/2203.04586
- Task: Missing-modality synthesis and brain tumor segmentation.
- Classification or segmentation: Segmentation.
- Graph / hypergraph / missing-modality / fusion: Modality completion + modality-level attention fusion.
- Why I should read it: It directly targets missing T1ce synthesis, which is clinically close to your no-T1ce problem.
- Key idea: MAF-Net synthesizes unavailable modalities using patchwise contrastive learning and dynamically fuses modalities with attention.
- Possible inspiration for my method: Compare two choices experimentally: synthesize missing T1ce features versus leave T1ce absent and let a mask-aware ROI hypergraph adapt.
- What is not applicable to my work: Pixel-level synthesis can hallucinate information and is more expensive than latent ROI completion.
- Priority: Must read

## A Multimodal Feature Distillation with CNN-Transformer Network for Brain Tumor Segmentation with Incomplete Modalities

- Year / venue: 2024 / arXiv. Source: https://arxiv.org/abs/2404.14019
- Task: Brain tumor segmentation with incomplete modalities.
- Classification or segmentation: Segmentation.
- Graph / hypergraph / missing-modality / fusion: Multimodal feature distillation + cross-modal fusion.
- Why I should read it: It combines feature distillation with explicit cross-modal fusion, which is useful for designing a mid-level ROI feature objective.
- Key idea: MCTSeg distills multimodal knowledge into unimodal features, enhances local/global unimodal context, and aligns global cross-modal correlations.
- Possible inspiration for my method: Distill a full-modality ROI representation into single- or two-modality ROI representations before hypergraph propagation.
- What is not applicable to my work: The CNN-Transformer segmentation architecture may be unnecessary if your main novelty is graph-structured classification.
- Priority: Useful

## Unveiling Incomplete Modality Brain Tumor Segmentation: Leveraging Masked Predicted Auto-Encoder and Divergence Learning

- Year / venue: 2024 / arXiv. Source: https://arxiv.org/abs/2406.08634
- Task: Missing-modality brain tumor segmentation.
- Classification or segmentation: Segmentation.
- Graph / hypergraph / missing-modality / fusion: Masked predicted pretraining + divergence-based distillation.
- Why I should read it: It is useful for thinking about alternatives to KL divergence and how to align complete and incomplete features.
- Key idea: Masked predicted pretraining improves incomplete-modality feature learning, and Holder pseudo-divergence guides distillation during fine-tuning.
- Possible inspiration for my method: Test a divergence or contrastive objective between full-modality and missing-modality ROI embeddings.
- What is not applicable to my work: The paper is not graph-based and focuses on segmentation feature maps.
- Priority: Useful

## Mind the Gap: Promoting Missing Modality Brain Tumor Segmentation with Alignment

- Year / venue: 2024 / arXiv. Source: https://arxiv.org/abs/2409.19366
- Task: Missing-modality brain tumor segmentation.
- Classification or segmentation: Segmentation.
- Graph / hypergraph / missing-modality / fusion: Feature alignment + teacher-student learning.
- Why I should read it: It gives a clean conceptual language for the complete-modality versus missing-modality feature gap.
- Key idea: Latent modality features are aligned to a distribution anchor, improving teacher guidance and missing-modality student performance.
- Possible inspiration for my method: Define full-modality ROI anchors and align missing-modality ROI embeddings toward them without reconstructing images.
- What is not applicable to my work: Distribution-anchor alignment alone does not explain anatomical ROI relations.
- Priority: Useful

## Set-Inclusive Uncertainty Modeling for Robust Brain Tumor Segmentation

- Year / venue: 2026 / arXiv. Source: https://arxiv.org/abs/2606.30374
- Task: Missing-modality brain tumor segmentation.
- Classification or segmentation: Segmentation.
- Graph / hypergraph / missing-modality / fusion: Missing-modality uncertainty + modality-subset ordering.
- Why I should read it: It is valuable if you want your model to report confidence degradation when stronger modalities are absent.
- Key idea: Representations are modeled probabilistically, and variance is regularized to reflect information deficiency across modality subsets.
- Possible inspiration for my method: Add uncertainty estimates per modality subset or per ROI, especially for single-modality and no-T1ce cases.
- What is not applicable to my work: It is very recent and segmentation-focused; uncertainty modeling may complicate the first version of your classifier.
- Priority: Useful

## Modality Completion via Gaussian Process Prior Variational Autoencoders for Multi-Modal Glioma Segmentation

- Year / venue: 2021 / arXiv. Source: https://arxiv.org/abs/2107.03442
- Task: Missing-modality completion for glioma segmentation.
- Classification or segmentation: Segmentation.
- Graph / hypergraph / missing-modality / fusion: Modality completion + generative modeling.
- Why I should read it: It is a focused modality-completion paper and helps separate image-level imputation from feature-level missing-modality handling.
- Key idea: MGP-VAE imputes missing MRI sub-modalities by using a Gaussian Process prior to model patient and modality correlations.
- Possible inspiration for my method: Use patient/modality correlation as a prior for latent ROI feature completion, rather than generating full images.
- What is not applicable to my work: Full modality imputation may be too indirect for classification and may introduce synthetic evidence.
- Priority: Background

## AMM-Diff: Adaptive Multi-Modality Diffusion Network for Missing Modality Imputation

- Year / venue: 2025 / arXiv. Source: https://arxiv.org/abs/2501.12840
- Task: Missing MRI modality imputation for downstream brain tumor segmentation.
- Classification or segmentation: Segmentation support via imputation.
- Graph / hypergraph / missing-modality / fusion: Adaptive modality completion + diffusion.
- Why I should read it: It is a recent generative-completion method that can handle varying input modality configurations.
- Key idea: AMM-Diff uses adaptive diffusion and image-frequency fusion to generate missing modalities from available ones.
- Possible inspiration for my method: If you later add completion, make it adaptive to arbitrary modality subsets rather than training one completion model per missing pattern.
- What is not applicable to my work: Diffusion-based image completion is likely too expensive and too far from the current ROI hypergraph classifier.
- Priority: Background

## Improved mmFormer for Liver Fibrosis Staging via Missing-Modality Compensation

- Year / venue: 2025 / arXiv. Source: https://arxiv.org/abs/2509.16436
- Task: Missing-modality multimodal MRI classification for liver fibrosis staging.
- Classification or segmentation: Classification.
- Graph / hypergraph / missing-modality / fusion: Missing-modality compensation + modality masks + adaptive feature synthesis.
- Why I should read it: It is not brain tumor work, but it is directly relevant to missing-modality MRI classification, which is rare in the brain tumor literature.
- Key idea: The model adapts mmFormer-style modality encoders to classification and uses zero-padding, availability masks, and learnable statistical compensation for missing modalities.
- Possible inspiration for my method: Encode modality availability explicitly and test whether learnable proxy features help or hurt compared with strict mask-aware observed-modality fusion.
- What is not applicable to my work: The disease, modalities, and task are different; use it only as a transfer idea for classification under missing MRI inputs.
- Priority: Background

## Multimodal Brain Tumour Classification Using Feature Fusion

- Year / venue: 2026 / arXiv. Source: https://arxiv.org/abs/2606.11107
- Task: Brain tumor classification using MRI and radiomic features.
- Classification or segmentation: Classification.
- Graph / hypergraph / missing-modality / fusion: Adaptive feature fusion, but not missing-modality specific.
- Why I should read it: It is classification-focused and compares concatenation, gating, and bidirectional cross-modal attention.
- Key idea: A two-branch model fuses CNN image features and radiomic features; gated fusion performs best in the reported experiments.
- Possible inspiration for my method: Compare simple concatenation, gated fusion, attention fusion, and ROI hypergraph fusion as classification heads.
- What is not applicable to my work: It is not designed for missing MRI sequences and does not model anatomical ROI relations.
- Priority: Background

# First 8 Papers To Read

1. ReHyDIL: hypergraph + missing modalities + brain tumor MRI.
2. D3Seg: modality graph fusion + latent diffusion completion.
3. mmFormer: canonical incomplete multimodal Transformer baseline.
4. M3AE: modality dropout, masked autoencoding, and self-distillation.
5. AdaMM: graph-guided distillation and practical missing-modality strategy comparison.
6. DIGEST: clean teacher-student setup for incomplete modalities.
7. ACN: missing modality as feature/domain gap.
8. MAF-Net: T1ce synthesis and modality-level attention fusion.

# Recommended Reading Order

1. Start with mmFormer to understand the standard incomplete multimodal segmentation problem.
2. Read M3AE to understand modality dropout, masked reconstruction, and catch-all single-model training.
3. Read DIGEST and ACN together to compare teacher-student transfer versus adversarial co-training.
4. Read AdaMM to see a recent distillation-heavy design with graph-guided refinement.
5. Read ReHyDIL to understand the closest hypergraph/missing-modality overlap.
6. Read D3Seg to think about modality graphs and latent feature completion.
7. Read MAF-Net and MGP-VAE to decide whether image/feature completion is worth adding.
8. Read the graph/CNN segmentation and classification fusion papers last for design alternatives.

# Three Method Ideas Inspired By These Papers

1. **ROI-level full-to-missing distillation**: train a full-modality teacher and distill logits plus ROI embeddings into the same model under sampled missing-modality masks.
2. **Mask-aware ROI hypergraph with optional latent T1ce completion**: keep observed-modality fusion as the base model, then add a small T1ce ROI-feature predictor and test it only on no-T1ce patterns.
3. **Modality-subset reliability head**: predict uncertainty or reliability from the availability mask, ROI embeddings, and graph-edge responses so weak modality combinations are flagged instead of overconfidently classified.

# Three Risks Or Traps To Avoid

1. **Do not claim hypergraph novelty too broadly**: ReHyDIL already uses hypergraphs for missing-modality brain tumor segmentation. Your distinction should be classification, within-case ROI structure, and mask-aware modality aggregation.
2. **Do not add image synthesis unless it proves useful**: completion papers are inspiring, but synthetic modalities can introduce hallucinated evidence and may distract from the cleaner ROI hypergraph story.
3. **Do not compare modality subsets unfairly**: use the same split, checkpoint-selection rule, calibration protocol, and evaluation metrics across full-modality, no-T1ce, single-modality, and mixed-modality cases.

# 中文翻译

范围：这份清单用于找方法灵感，不是完整文献综述。重点是能启发“缺失模态多模态脑 MRI 分类 + ROI 图/超图推理”的论文。部分论文是 arXiv 预印本，正式引用前需要再次确认最终发表信息。

## Hypergraph Tversky-Aware Domain Incremental Learning for Brain Tumor Segmentation with Missing Modalities

- 年份 /  venue：2025 / arXiv。来源：https://arxiv.org/abs/2505.16809
- 任务：缺失模态脑肿瘤分割。
- 分类或分割：分割。
- 图 / 超图 / 缺失模态 / 融合：超图 + 缺失模态 + 域增量学习。
- 为什么要读：这是最接近你研究空间的论文之一，因为它同时涉及超图、缺失 MRI 模态和脑肿瘤。
- 核心思想：ReHyDIL 用跨患者超图分割网络建模高阶患者关系，并用 Tversky-aware contrastive loss 缓解模态不平衡。
- 对我的方法的启发：可以借鉴“用超边表达高阶关系”的思路，但把关系改成病例内部的 ROI-模态关系，而不是跨患者分割关系。
- 不适用于我的地方：它是分割和域增量学习，不是 HGG/LGG 分类；跨患者超图也不同于解剖 ROI 超图。
- 优先级：Must read

## D3Seg: Dependency-Aware Diffusion for Brain Tumor Segmentation with Missing Modalities

- 年份 / venue：2026 / arXiv。来源：https://arxiv.org/abs/2605.22249
- 任务：缺失模态脑肿瘤分割。
- 分类或分割：分割。
- 图 / 超图 / 缺失模态 / 融合：模态图融合 + 潜在空间 diffusion 补全 + 缺失模态鲁棒性。
- 为什么要读：它把模态依赖图和轻量补全结合起来，非常适合启发你的 T1ce latent feature completion 方向。
- 核心思想：Multi-hop Modality Graph Fusion 建模高阶模态依赖，diffusion latent imputation 补偿缺失 T1ce 表示。
- 对我的方法的启发：在 ROI 编码后加一个小的 latent completion head，测试预测的 T1ce ROI 特征能否提升 no-T1ce 设置，同时不损害全模态性能。
- 不适用于我的地方：它仍然是分割方法，关注 dense prediction 的潜在补全，不是分类校准。
- 优先级：Must read

## A Joint Graph and Image Convolution Network for Automatic Brain Tumor Segmentation

- 年份 / venue：2021 / arXiv，BraTS 2021 challenge method。来源：https://arxiv.org/abs/2109.05580
- 任务：脑肿瘤分割。
- 分类或分割：分割。
- 图 / 超图 / 缺失模态 / 融合：图学习 + CNN 细化。
- 为什么要读：它是一个简单清晰的例子，把每个脑 MRI 表示成图结构区域，再用 CNN 细化。
- 核心思想：先用 GNN 对图定义的脑区域进行分割，再用 voxel CNN 细化肿瘤预测。
- 对我的方法的启发：先把图节点作为可解释的解剖/ROI 单元，再做分类，而不是把整个 3D 图像当作黑盒特征。
- 不适用于我的地方：不处理缺失模态，也不是超图。
- 优先级：Useful

## No Modality Left Behind: Adapting to Missing Modalities via Knowledge Distillation for Brain Tumor Segmentation

- 年份 / venue：2025 / arXiv。来源：https://arxiv.org/abs/2509.15017
- 任务：缺失模态脑肿瘤分割。
- 分类或分割：分割，包含辅助 lesion-presence 分类信号。
- 图 / 超图 / 缺失模态 / 融合：图引导自适应细化 + teacher-student 蒸馏 + 可靠性建模。
- 为什么要读：这是一篇较新的缺失模态策略论文，并系统比较了多类缺失模态方法。
- 核心思想：AdaMM 结合 graph-guided adaptive refinement、bi-bottleneck distillation 和 lesion-presence-guided reliability。
- 对我的方法的启发：用全模态 teacher 正则化缺失模态下的 ROI embedding；如果能改善校准，可加入辅助可靠性或 lesion-presence proxy。
- 不适用于我的地方：dense segmentation distillation 对紧凑分类模型可能太重。
- 优先级：Must read

## mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation

- 年份 / venue：2022 / arXiv。来源：https://arxiv.org/abs/2206.02425
- 任务：不完整多模态脑肿瘤分割。
- 分类或分割：分割。
- 图 / 超图 / 缺失模态 / 融合：缺失模态 + intra-modal / inter-modal Transformer 融合。
- 为什么要读：这是缺失模态分割的核心基线之一，展示了如何用模态特异编码器处理任意模态子集。
- 核心思想：CNN/Transformer 混合编码器建模单模态上下文，inter-modal Transformer 对齐跨模态全局语义。
- 对我的方法的启发：区分模态特异特征提取和跨模态交互；把全局 token attention 换成 ROI-level modality hyperedges。
- 不适用于我的地方：Transformer dense prediction 更大，也不如 ROI 超图分类器解剖结构明确。
- 优先级：Must read

## M3AE: Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities

- 年份 / venue：2023 / arXiv。来源：https://arxiv.org/abs/2303.05302
- 任务：缺失模态脑肿瘤分割。
- 分类或分割：分割。
- 图 / 超图 / 缺失模态 / 融合：modality dropout + masked autoencoding + self-distillation。
- 为什么要读：它非常适合用来理解 modality dropout 和鲁棒预训练。
- 核心思想：预训练时随机 mask 模态和 patch；model inversion 与 self-distillation 用来适配不同缺失模态情况。
- 对我的方法的启发：在监督 HGG/LGG 分类前，用 modality dropout 或 masked ROI-feature reconstruction 预训练 ROI encoder。
- 不适用于我的地方：如果模型本身只基于可观测模态推理，图像重建/替代不一定必要。
- 优先级：Must read

## ACN: Adversarial Co-training Network for Brain Tumor Segmentation with Missing Modalities

- 年份 / venue：2021 / arXiv。来源：https://arxiv.org/abs/2106.14591
- 任务：缺失模态脑肿瘤分割。
- 分类或分割：分割。
- 图 / 超图 / 缺失模态 / 融合：缺失模态 + adversarial co-training + latent alignment。
- 为什么要读：它是把缺失模态视为 full/incomplete 输入之间 domain gap 的早期代表。
- 核心思想：全模态模型和缺失模态模型协同训练，并用对抗损失对齐 domain、增强预测可靠性。
- 对我的方法的启发：用轻量 contrastive 或 adversarial regularizer 对齐全模态与缺失模态 ROI embedding。
- 不适用于我的地方：为每种缺失情况训练独立模型不适合一个统一分类器目标。
- 优先级：Must read

## DIGEST: Deeply Supervised Knowledge Transfer Network Learning for Brain Tumor Segmentation with Incomplete Multi-modal MRI Scans

- 年份 / venue：2022 / arXiv。来源：https://arxiv.org/abs/2211.07993
- 任务：不完整多模态 MRI 脑肿瘤分割。
- 分类或分割：分割。
- 图 / 超图 / 缺失模态 / 融合：缺失模态 + teacher-student 学习。
- 为什么要读：它给出了非常清晰的 teacher-student 公式，用全模态知识监督缺失模态 student。
- 核心思想：student 在 Bernoulli 采样的缺失模态条件下训练，并从全模态 teacher 的多个 decoder stage 学习语义信息。
- 对我的方法的启发：把全模态 classifier logits 和 ROI embedding 蒸馏给随机模态子集下的同一个 classifier。
- 不适用于我的地方：decoder-stage dense supervision 不能直接迁移到病例级分类。
- 优先级：Must read

## Multi-modal Brain Tumor Segmentation via Missing Modality Synthesis and Modality-level Attention Fusion

- 年份 / venue：2022 / arXiv。来源：https://arxiv.org/abs/2203.04586
- 任务：缺失模态合成与脑肿瘤分割。
- 分类或分割：分割。
- 图 / 超图 / 缺失模态 / 融合：模态补全 + 模态级注意力融合。
- 为什么要读：它直接处理 T1ce 合成，与你的 no-T1ce 问题很接近。
- 核心思想：MAF-Net 用 patchwise contrastive learning 合成不可用模态，并用 attention 动态融合不同模态。
- 对我的方法的启发：可以实验比较“合成缺失 T1ce 特征”和“保持 T1ce 缺失，让 mask-aware ROI hypergraph 自适应”这两条路线。
- 不适用于我的地方：像素级合成可能产生幻觉信息，也比 latent ROI completion 更昂贵。
- 优先级：Must read

## A Multimodal Feature Distillation with CNN-Transformer Network for Brain Tumor Segmentation with Incomplete Modalities

- 年份 / venue：2024 / arXiv。来源：https://arxiv.org/abs/2404.14019
- 任务：不完整模态脑肿瘤分割。
- 分类或分割：分割。
- 图 / 超图 / 缺失模态 / 融合：多模态特征蒸馏 + 跨模态融合。
- 为什么要读：它把 feature distillation 和 cross-modal fusion 结合起来，适合启发 ROI 中间特征目标。
- 核心思想：MCTSeg 把多模态知识蒸馏到单模态特征中，增强单模态局部/全局上下文，并对齐跨模态全局相关性。
- 对我的方法的启发：把全模态 ROI 表示蒸馏到单模态或双模态 ROI 表示，再进行超图传播。
- 不适用于我的地方：CNN-Transformer 分割架构可能对你的图结构分类主线来说过重。
- 优先级：Useful

## Unveiling Incomplete Modality Brain Tumor Segmentation: Leveraging Masked Predicted Auto-Encoder and Divergence Learning

- 年份 / venue：2024 / arXiv。来源：https://arxiv.org/abs/2406.08634
- 任务：缺失模态脑肿瘤分割。
- 分类或分割：分割。
- 图 / 超图 / 缺失模态 / 融合：masked predicted pretraining + divergence-based distillation。
- 为什么要读：它适合理解 KL 之外的 divergence loss，以及如何对齐 complete 和 incomplete feature。
- 核心思想：masked predicted pretraining 提高不完整模态特征学习能力，Holder pseudo-divergence 在 fine-tuning 阶段指导蒸馏。
- 对我的方法的启发：在 full-modality 与 missing-modality ROI embedding 之间测试 divergence 或 contrastive objective。
- 不适用于我的地方：它不是 graph-based，主要处理 segmentation feature maps。
- 优先级：Useful

## Mind the Gap: Promoting Missing Modality Brain Tumor Segmentation with Alignment

- 年份 / venue：2024 / arXiv。来源：https://arxiv.org/abs/2409.19366
- 任务：缺失模态脑肿瘤分割。
- 分类或分割：分割。
- 图 / 超图 / 缺失模态 / 融合：特征对齐 + teacher-student learning。
- 为什么要读：它很好地解释了 full-modality 和 missing-modality feature gap。
- 核心思想：把 latent modality features 对齐到 distribution anchor，从而改善 teacher guidance 和 missing-modality student 表现。
- 对我的方法的启发：定义 full-modality ROI anchors，让 missing-modality ROI embeddings 向这些 anchors 对齐，而不是重建图像。
- 不适用于我的地方：单纯 distribution-anchor alignment 不解释解剖 ROI 关系。
- 优先级：Useful

## Set-Inclusive Uncertainty Modeling for Robust Brain Tumor Segmentation

- 年份 / venue：2026 / arXiv。来源：https://arxiv.org/abs/2606.30374
- 任务：缺失模态脑肿瘤分割。
- 分类或分割：分割。
- 图 / 超图 / 缺失模态 / 融合：缺失模态不确定性 + 模态子集顺序约束。
- 为什么要读：如果你希望模型在关键模态缺失时输出可信度下降，它很有参考价值。
- 核心思想：把表示建模为概率分布，并让 variance 反映不同模态子集的信息缺失程度。
- 对我的方法的启发：给每个模态组合或每个 ROI 加 uncertainty estimate，尤其关注 single-modality 和 no-T1ce 设置。
- 不适用于我的地方：它非常新且面向分割；不确定性建模可能让第一版分类器变复杂。
- 优先级：Useful

## Modality Completion via Gaussian Process Prior Variational Autoencoders for Multi-Modal Glioma Segmentation

- 年份 / venue：2021 / arXiv。来源：https://arxiv.org/abs/2107.03442
- 任务：胶质瘤分割中的缺失模态补全。
- 分类或分割：分割。
- 图 / 超图 / 缺失模态 / 融合：模态补全 + 生成模型。
- 为什么要读：它能帮助你区分图像级补全和特征级缺失模态处理。
- 核心思想：MGP-VAE 使用 Gaussian Process prior 建模患者和模态相关性，从而补全缺失 MRI 子模态。
- 对我的方法的启发：把 patient/modality correlation 作为 latent ROI feature completion 的先验，而不是生成完整图像。
- 不适用于我的地方：完整模态 imputation 对分类可能太间接，也可能引入合成证据。
- 优先级：Background

## AMM-Diff: Adaptive Multi-Modality Diffusion Network for Missing Modality Imputation

- 年份 / venue：2025 / arXiv。来源：https://arxiv.org/abs/2501.12840
- 任务：缺失 MRI 模态补全，用于支持后续脑肿瘤分割。
- 分类或分割：为分割服务的 imputation。
- 图 / 超图 / 缺失模态 / 融合：自适应模态补全 + diffusion。
- 为什么要读：它是较新的生成式补全方法，可以处理变化的输入模态组合。
- 核心思想：AMM-Diff 使用 adaptive diffusion 和 image-frequency fusion，从可用模态生成缺失模态。
- 对我的方法的启发：如果后续加入 completion，模块应适配任意模态子集，而不是只适配一种缺失模式。
- 不适用于我的地方：diffusion 图像补全很可能计算昂贵，也偏离当前 ROI hypergraph classifier 主线。
- 优先级：Background

## Improved mmFormer for Liver Fibrosis Staging via Missing-Modality Compensation

- 年份 / venue：2025 / arXiv。来源：https://arxiv.org/abs/2509.16436
- 任务：缺失模态多模态 MRI 肝纤维化分期分类。
- 分类或分割：分类。
- 图 / 超图 / 缺失模态 / 融合：缺失模态补偿 + 模态 mask + 自适应特征合成。
- 为什么要读：它不是脑肿瘤论文，但直接对应“缺失模态 MRI 分类”，而脑肿瘤分类方向这类论文较少。
- 核心思想：把 mmFormer 式模态编码器改造成分类模型，并用 zero-padding、availability mask 和可学习统计补偿处理缺失模态。
- 对我的方法的启发：显式编码 modality availability，并测试 learnable proxy features 相比严格 mask-aware observed-modality fusion 是否有收益。
- 不适用于我的地方：疾病、模态和任务不同；只能作为 MRI 分类缺失模态设计的迁移灵感。
- 优先级：Background

## Multimodal Brain Tumour Classification Using Feature Fusion

- 年份 / venue：2026 / arXiv。来源：https://arxiv.org/abs/2606.11107
- 任务：使用 MRI 与 radiomic features 的脑肿瘤分类。
- 分类或分割：分类。
- 图 / 超图 / 缺失模态 / 融合：自适应特征融合，但不是缺失模态方法。
- 为什么要读：它聚焦分类，并比较 concatenation、gating、bidirectional cross-modal attention。
- 核心思想：双分支模型融合 CNN 图像特征与 radiomic features；文中 gated fusion 效果最好。
- 对我的方法的启发：把简单 concat、gated fusion、attention fusion 和 ROI hypergraph fusion 作为分类头对比。
- 不适用于我的地方：它不面向缺失 MRI 序列，也不建模解剖 ROI 关系。
- 优先级：Background

## 最先读的 8 篇

1. ReHyDIL：超图 + 缺失模态 + 脑肿瘤 MRI。
2. D3Seg：模态图融合 + latent diffusion completion。
3. mmFormer：不完整多模态 Transformer 标准基线。
4. M3AE：modality dropout、masked autoencoding、自蒸馏。
5. AdaMM：图引导蒸馏与缺失模态策略比较。
6. DIGEST：清晰的 teacher-student 缺失模态框架。
7. ACN：把缺失模态建模为 feature/domain gap。
8. MAF-Net：T1ce 合成与模态级注意力融合。

## 推荐阅读顺序

1. 先读 mmFormer，建立缺失模态分割问题的标准框架。
2. 再读 M3AE，理解 modality dropout、masked reconstruction 和 catch-all 单模型训练。
3. 把 DIGEST 和 ACN 放在一起读，比较 teacher-student transfer 与 adversarial co-training。
4. 读 AdaMM，看较新的蒸馏式设计和 graph-guided refinement。
5. 读 ReHyDIL，理解最接近的 hypergraph/missing-modality 交叉点。
6. 读 D3Seg，思考 modality graph 与 latent feature completion。
7. 读 MAF-Net 和 MGP-VAE，判断是否值得加入 image/feature completion。
8. 最后读 graph/CNN segmentation 与 classification fusion 论文，作为结构替代方案。

## 三个可尝试的方法想法

1. **ROI-level full-to-missing distillation**：训练全模态 teacher，把 logits 和 ROI embedding 蒸馏到随机缺失模态输入下的同一模型。
2. **Mask-aware ROI hypergraph + optional latent T1ce completion**：先保留可观测模态融合为主模型，再只在 no-T1ce 设置下测试一个小型 T1ce ROI-feature predictor。
3. **Modality-subset reliability head**：根据 availability mask、ROI embedding 和 graph-edge response 预测不确定性/可靠性，避免弱模态组合下过度自信。

## 三个要避免的风险

1. **不要过度声称超图首创性**：ReHyDIL 已经把超图用于缺失模态脑肿瘤分割。你的区别应放在分类、病例内部 ROI 结构、mask-aware modality aggregation。
2. **不要为了追热点加入图像合成**：completion 论文很有启发，但合成模态可能引入伪证据，也会削弱 ROI hypergraph 的清晰叙事。
3. **不要让不同模态组合比较不公平**：full-modality、no-T1ce、single-modality、mixed-modality 必须使用相同 split、checkpoint selection、calibration protocol 和评价指标。
