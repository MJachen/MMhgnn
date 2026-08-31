# Closest Papers And Positioning

Source: `literature_matrix.csv`. This file ranks papers by closeness to the project and explains how to position against them.

## Project Anchor

The local project is a BraTS-oriented HGG/LGG classification model for missing-modality multimodal MRI. Its current design is a two-stage hybrid dynamic hypergraph:

1. **Within each ROI**: aggregate currently available modalities using local modality hyperedges.
2. **Across shared ROI nodes**: combine fixed medical-prior hyperedges with patient-specific dynamic KNN hyperedges.

The closest papers are therefore not simply the strongest segmentation models. They are papers that overlap on missing modalities, multimodal MRI, hypergraph or graph structure, and brain tumor decision modeling.

## Tier 1: Directly Closest

### `MM-2025-REHYDIL`

**Why it is close**

- It explicitly combines hypergraph modeling, missing modalities, and brain tumor MRI.
- It uses a cross-patient hypergraph segmentation network and domain-incremental learning.
- It is the only matrix entry that strongly overlaps all three major keywords: hypergraph, missing modality, and brain tumor MRI.

**Key difference**

- ReHyDIL is a segmentation framework under domain-incremental learning.
- This project is patient-level HGG/LGG classification with a within-case two-stage ROI hypergraph.
- ReHyDIL uses hypergraphs for cross-patient high-order associations; this project uses hypergraphs to model modality aggregation inside ROIs and anatomical/dynamic relations across ROIs.

**Positioning sentence**

ReHyDIL shows that hypergraph modeling is beginning to enter missing-modality brain tumor segmentation, but its cross-patient domain-incremental setting differs from the proposed within-case ROI hypergraph for missing-modality glioma classification.

## Tier 2: Closest Missing-Modality Baselines

### `MM-2022-MMFORMER`

**Why it is close**

- It targets incomplete multimodal brain tumor segmentation.
- It handles arbitrary available modality subsets with modality-specific encoders and inter-modal Transformer modeling.

**Key difference**

- mmFormer uses transformer attention for incomplete multimodal segmentation.
- This project uses explicit ROI and hyperedge structure for classification.

**Positioning angle**

Use mmFormer as the representative transformer-based incomplete multimodal learning baseline. The contrast is attention-based global feature alignment versus anatomy-guided hypergraph reasoning.

### `MM-2023-M3AE`

**Why it is close**

- It directly addresses missing modalities in multimodal brain tumor segmentation.
- It uses masked autoencoding, model inversion, and self-distillation.
- It is relevant to any future latent-feature completion extension.

**Key difference**

- M3AE reconstructs or substitutes missing-modality information.
- This project should currently be framed as observed-modality, mask-aware ROI fusion unless completion is experimentally added.

**Positioning angle**

Use M3AE to distinguish reconstruction-heavy missing-modality learning from structure-preserving observed-modality fusion.

### `MM-2021-ACN`

**Why it is close**

- It is a strong early missing-modality brain tumor segmentation baseline.
- It frames missing modalities as a domain and feature-representation gap.

**Key difference**

- ACN trains related full/missing modality models with adversarial co-training.
- This project seeks one compact ROI-hypergraph classifier that can operate under modality subsets.

**Positioning angle**

Use ACN as the classic co-training/adversarial missing-modality baseline and contrast it with explicit anatomical relational modeling.

### `MM-2022-DIGEST`

**Why it is close**

- It uses teacher-student knowledge transfer for incomplete multimodal MRI segmentation.
- It simulates missing conditions during training.

**Key difference**

- DIGEST transfers dense segmentation knowledge from full to incomplete modalities.
- This project uses missingness-aware ROI graph construction for classification.

**Positioning angle**

Use DIGEST to represent distillation-based missing-modality robustness.

## Tier 3: Relevant Recent Missing-Modality Directions

### `MM-2022-MAFNET`

MAF-Net is important for the synthesis/completion branch of the literature. It is useful when arguing that missing-modality work often tries to reconstruct absent sequences, while the current project can avoid image-level synthesis and reason over observed ROI evidence.

### `MM-2024-MPPA`

MPPA is relevant for masked pretraining and divergence-based distillation. It supports the observation that recent missing-modality work is moving toward pretraining and complete/incomplete feature alignment.

### `MM-2024-ALIGN`

This paper is useful for the modality-gap framing. It gives language for why complete-modality and missing-modality representations diverge, but it does not provide explicit anatomical hypergraph reasoning.

### `MM-2025-ADAMM`

AdaMM is a recent graph-guided distillation paper. It is relevant because it moves beyond purely convolutional/transformer fusion, but its graph-guided refinement remains different from a two-stage ROI hypergraph classifier.

### `MM-2026-SIUM`

SIUM is useful for reliability and uncertainty positioning. It is especially relevant if the paper discusses confidence under single-modality or weak-modality settings. Treat it cautiously because it is very recent.

## Tier 4: Segmentation Backbone Context

### `BTS-2021-TRANSBTS` and `BTS-2022-SWINUNETR`

These are not closest in problem setting because they are not primarily missing-modality works. They are important because they represent the CNN/Transformer segmentation backbone landscape that missing-modality methods build on.

Positioning use:

- Cite or discuss them as context for multimodal BraTS segmentation architectures.
- Do not frame them as direct missing-modality competitors.

### `BTS-2021-REDUNDANCY` and `BTS-2022-MULTICNN`

These are challenge-performance context papers. They are useful for showing the maturity of BraTS segmentation pipelines and ensembles, not for defining the main novelty gap.

### `BTS-2026-HYPERCONN`

This is adjacent because it studies adaptive multimodal fusion and reports modality sensitivity. It should not be confused with hypergraph learning. Its value is in showing that adaptive fusion remains an active issue in multimodal brain tumor segmentation.

## Tier 5: Hypergraph Method Foundations

### `HYP-2021-UNIGNN`

Best for explaining the general move from graph message passing to hypergraph message passing.

### `HYP-2021-ALLSET`

Best for explaining learnable set/multiset aggregation over hyperedges.

### `HYP-2022-EDHNN`

Best for diffusion-style and irregular higher-order relation modeling.

### `HYP-2023-HYPERGT`

Best for global node-hyperedge interaction and transformer-style hypergraph modeling.

### `HYP-2025-REVIEW`

Best for taxonomy support, but it should be cited carefully until final publication status is checked.

## Closest-Paper Ranking

| Rank | Paper ID | Closeness | Main reason |
|---:|---|---|---|
| 1 | `MM-2025-REHYDIL` | Highest | Hypergraph + missing modalities + brain tumor MRI, though segmentation/domain-incremental. |
| 2 | `MM-2022-MMFORMER` | High | Incomplete multimodal brain tumor segmentation with transformer-based modality interaction. |
| 3 | `MM-2023-M3AE` | High | Missing-modality representation learning and reconstruction/substitution. |
| 4 | `MM-2021-ACN` | High | Classic adversarial co-training baseline for missing-modality BraTS segmentation. |
| 5 | `MM-2022-DIGEST` | Medium-high | Distillation-based incomplete multimodal segmentation. |
| 6 | `MM-2025-ADAMM` | Medium-high | Graph-guided missing-modality distillation, recent and relevant. |
| 7 | `MM-2026-SIUM` | Medium | Missing-evidence uncertainty modeling. |
| 8 | `BTS-2026-HYPERCONN` | Medium | Adaptive multimodal fusion, but not hypergraph or missing-modality focused. |
| 9 | `HYP-2021-UNIGNN` / `HYP-2021-ALLSET` | Method foundation | Hypergraph theory foundation, not MRI-specific. |
| 10 | `BTS-2021-TRANSBTS` / `BTS-2022-SWINUNETR` | Context | Strong multimodal segmentation backbones, not direct competitors. |

## Recommended Narrative Boundary

The safest narrative is:

- Do not claim the method is the first hypergraph model in missing-modality brain tumor MRI.
- Claim that the closest known hypergraph-overlap work is segmentation-focused and cross-patient/domain-incremental.
- Claim that most missing-modality MRI methods rely on synthesis, distillation, alignment, attention, or uncertainty rather than an explicit two-stage ROI hypergraph for classification.
- Claim that the project contributes a structurally interpretable classification route: modality-aware ROI aggregation followed by prior-guided and patient-adaptive ROI relation modeling.

