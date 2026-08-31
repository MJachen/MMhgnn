# Literature Taxonomy

Source: `literature_matrix.csv` generated on 2026-07-06. This taxonomy is for research positioning, not a paper-by-paper summary.

## Positioning Axes

The reviewed papers separate along three axes that matter for this project:

1. **Task target**: voxel-level tumor segmentation, HGG/LGG or glioma classification, or generic hypergraph learning.
2. **Missing-modality strategy**: ignore missingness, synthesize/reconstruct missing modalities, distill from complete-modality teachers, align latent features, model uncertainty, or build mask-aware representations directly.
3. **Relational structure**: no explicit graph, graph-guided refinement, general HGNN theory, or MRI-specific hypergraph modeling.

The project sits in the under-populated intersection: **missing-modality multimodal brain MRI classification with explicit ROI-level hypergraph reasoning**.

## Taxonomy Table

| Family | Representative papers | Core idea | What it explains well | What it does not cover for this project |
|---|---|---|---|---|
| General hypergraph learning | `HYP-2021-UNIGNN`, `HYP-2021-ALLSET`, `HYP-2022-EDHNN`, `HYP-2023-HYPERGT`, `HYP-2025-REVIEW` | Learn over higher-order relations using hypergraph message passing, multiset functions, diffusion, or transformer-style node-hyperedge interaction. | Why hypergraphs are a reasonable modeling language for non-pairwise dependencies among ROIs and modalities. | These works are not brain MRI methods and do not solve missing-modality clinical imaging directly. |
| Full-modality multimodal BraTS segmentation | `BTS-2021-TRANSBTS`, `BTS-2021-REDUNDANCY`, `BTS-2022-SWINUNETR`, `BTS-2022-MULTICNN` | Strong 3D CNN/Transformer/ensemble segmentation models for complete or challenge-style multimodal MRI. | Establishes the modern BraTS segmentation backbone landscape and the dominance of CNN/Transformer architectures. | Mostly voxel segmentation, not HGG/LGG classification; missing modalities are not the central modeling assumption. |
| Adaptive multimodal segmentation without explicit missing-modality objective | `BTS-2026-HYPERCONN` | Adaptive connections improve multimodal fusion across standard segmentation backbones. | Shows recent interest in adaptive multimodal fusion and modality sensitivity. | Hyper-connections are not hypergraphs; modality ablation is not equivalent to a full missing-modality protocol. |
| Glioma MRI classification and explainability | `CLS-2021-XAI` | Classification model improved by explanation-guided training. | Supports the relevance of classification and interpretability for glioma grading. | Does not address multimodal missingness or hypergraph-structured ROI reasoning. |
| Missing-modality segmentation via dedicated or co-trained models | `MM-2021-ACN` | Train full-modality and missing-modality models jointly, reduce feature/domain gaps adversarially. | Frames missing modalities as a domain/representation mismatch problem. | Mainly segmentation; dedicated missing-situation modeling can be heavier than one mask-aware classifier. |
| Missing-modality segmentation via synthesis/completion | `MM-2022-MAFNET`, partly `MM-2023-M3AE` | Synthesize missing modalities or substitute reconstructed/full-modal representations before downstream segmentation. | Motivates modality completion as one solution to absent T1ce or other missing sequences. | Image-level synthesis may add cost and error propagation; the project currently emphasizes observed-modality fusion and possible latent/ROI completion. |
| Missing-modality segmentation via Transformers and cross-modal alignment | `MM-2022-MMFORMER`, `MM-2024-ALIGN` | Use intra-modal and inter-modal attention, or align latent modality distributions to make representations invariant to missingness. | Strong baseline family for incomplete multimodal representation learning. | Relational structure is usually attention over feature tokens rather than interpretable ROI hyperedges. |
| Missing-modality segmentation via distillation | `MM-2022-DIGEST`, `MM-2024-MPPA`, `MM-2025-ADAMM` | Transfer knowledge from full-modality teachers to incomplete-modality students using feature/logit/style consistency. | Explains the main modern recipe for robust missing-modality segmentation. | Depends on segmentation teacher-student training and does not naturally expose ROI-level decision pathways for HGG/LGG classification. |
| Missing-modality segmentation via uncertainty or set structure | `MM-2026-SIUM` | Represent incomplete evidence probabilistically and impose ordering among modality subsets. | Useful for reliability framing under information loss. | Does not model anatomical ROI relations explicitly and is very recent preprint evidence. |
| Hypergraph plus missing-modality brain MRI | `MM-2025-REHYDIL` | Use domain-incremental learning and a cross-patient hypergraph segmentation network for missing-modality segmentation. | Closest thematic overlap: hypergraph, missing modalities, and brain tumor MRI. | Still segmentation, cross-patient/domain-incremental setting, not two-stage within-case ROI hypergraph classification. |

## Conceptual Map

```text
General HGNN theory
  -> higher-order relation modeling
  -> supports ROI/modal hyperedge design

BraTS segmentation backbones
  -> strong 3D CNN/Transformer baselines
  -> supports feature extraction context

Missing-modality segmentation
  -> synthesis, distillation, alignment, uncertainty
  -> supports robustness problem framing

Project position
  -> HGG/LGG classification
  -> missing-modality multimodal MRI
  -> two-stage ROI hypergraph:
     1. local modality hypergraph per ROI
     2. prior + dynamic KNN hypergraph across shared ROI nodes
```

## Taxonomic Claim

The matrix supports a three-part positioning claim:

- Hypergraph learning papers justify modeling higher-order relations, but they rarely instantiate those relations in multimodal brain MRI.
- Missing-modality brain tumor papers are mature for segmentation, but most solve robustness through synthesis, distillation, alignment, or uncertainty rather than explicit anatomical hypergraph reasoning.
- The project can be positioned as a task- and structure-specific bridge: it uses mask-aware multimodal features and ROI-level hypergraphs for missing-modality HGG/LGG classification, a narrower but more interpretable objective than general voxel segmentation.

