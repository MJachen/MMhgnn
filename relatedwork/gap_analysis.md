# Gap Analysis

Source: `literature_matrix.csv`. The goal is to identify research space, not to summarize each paper.

## Main Gap

The literature strongly covers two sides separately:

- **Hypergraph learning** provides tools for higher-order relation modeling, but mostly on generic node-classification benchmarks.
- **Missing-modality brain tumor MRI** provides robust segmentation methods, but mostly through synthesis, distillation, feature alignment, transformer attention, or uncertainty modeling.

The matrix contains only one close overlap paper, `MM-2025-REHYDIL`, and it is still a segmentation/domain-incremental framework. The specific space of **missing-modality multimodal brain MRI classification with explicit anatomical ROI hypergraphs** remains weakly covered.

## Gap 1: Segmentation Dominates, Classification Is Underdeveloped

Most brain tumor MRI papers in the matrix target voxel-level segmentation. This is expected because BraTS is segmentation-centered and has mature evaluation protocols. However, this project targets HGG/LGG classification from multimodal MRI under modality absence.

Research opportunity:

- Reframe missing-modality learning from dense prediction to patient-level decision support.
- Use segmentation-derived or anatomy-derived ROI nodes as an interpretable intermediate representation.
- Evaluate whether ROI-level relational features improve classification robustness when individual MRI sequences are missing.

Risk:

- Reviewers may ask why segmentation literature is used for a classification paper. The answer should be direct: segmentation methods define the multimodal and missing-modality feature-learning landscape, while the proposed contribution is decision-level ROI hypergraph classification.

## Gap 2: Missing-Modality Methods Often Treat Structure Implicitly

The main missing-modality families are:

- synthesis/completion: `MM-2022-MAFNET`, `MM-2023-M3AE`;
- distillation: `MM-2022-DIGEST`, `MM-2024-MPPA`, `MM-2025-ADAMM`;
- attention/alignment: `MM-2022-MMFORMER`, `MM-2024-ALIGN`;
- uncertainty: `MM-2026-SIUM`.

These methods are strong at representation transfer, but anatomical structure is usually implicit in convolutional features, transformer tokens, or teacher-student losses.

Research opportunity:

- Make anatomy an explicit computational object through ROI nodes and hyperedges.
- Separate within-ROI modality aggregation from across-ROI tumor-structure reasoning.
- Preserve interpretability through ROI importance, edge ablation, and modality-combination tests.

Risk:

- If the hypergraph is presented only as a generic architectural block, it may look like a weaker alternative to large transformer/distillation systems. The positioning should emphasize explicit structure, missing-modality ergonomics, and interpretability rather than raw architectural scale.

## Gap 3: Hypergraph Literature Is Method-Rich but Domain-Light

`HYP-2021-UNIGNN`, `HYP-2021-ALLSET`, `HYP-2022-EDHNN`, and `HYP-2023-HYPERGT` provide strong method concepts:

- graph-to-hypergraph message passing;
- learnable multiset aggregation;
- diffusion over irregular high-order relations;
- global node-hyperedge interaction.

But these papers do not answer clinical MRI questions:

- Which entities become nodes?
- Which anatomical or modality relations become hyperedges?
- How does the graph change when MRI modalities are missing?
- How should the model remain interpretable at patient level?

Research opportunity:

- Translate HGNN theory into a clinically meaningful hierarchy:
  - modality observations inside each ROI;
  - shared ROI representations;
  - prior anatomical hyperedges;
  - dynamic KNN hyperedges per patient.

Risk:

- Do not overclaim novelty over general HGNN methods. The novelty is not inventing HGNNs; it is applying a two-stage, mask-aware, anatomy-driven hypergraph to missing-modality brain tumor MRI classification.

## Gap 4: Missing-Modality Evaluation Needs Stronger Comparability

The missing-modality papers evaluate many modality subsets, but the protocols differ:

- some synthesize missing images;
- some train separate missing-condition models;
- some use teacher-student distillation;
- some test robustness to arbitrary subsets.

For this project, the key evaluation gap is not just average performance. It is whether the same model remains reliable across clinically plausible missing patterns.

Research opportunity:

- Report modality-combination performance explicitly, not only full-modality accuracy.
- Separate full-modality performance, no-T1ce performance, and single-modality stress cases.
- Use consistent checkpoint selection and identical splits so missing-modality claims are not artifacts of protocol drift.

Risk:

- A broad claim such as "robust to missing modalities" will be vulnerable unless supported across all relevant non-empty modality subsets or a clearly justified subset protocol.

## Gap 5: Completion vs Observed-Modality Fusion Is Still Unsettled

The matrix shows a spectrum:

- MAF-Net and M3AE lean toward synthesis or representative full-modal substitution.
- mmFormer and DIGEST learn robust incomplete representations.
- SIUM models uncertainty from missing evidence.
- The project currently uses observed-modality fusion with zero or mask-aware handling, with potential future latent/ROI completion.

Research opportunity:

- Position the current method as a transparent observed-modality fusion model first.
- Treat latent ROI completion as a possible extension, not as an already solved component.
- Compare against synthesis/completion papers by explaining the tradeoff: completion can recover information but may introduce hallucinated evidence, while mask-aware ROI hypergraphs preserve what is actually observed.

Risk:

- If completion language is used too strongly without experiments, it will overclaim. Keep current wording tied to observed-modality aggregation unless completion experiments are actually added.

## Best Research Positioning

The strongest positioning is:

> Existing missing-modality brain tumor MRI methods mainly improve segmentation robustness through synthesis, distillation, attention, alignment, or uncertainty. In parallel, hypergraph neural networks provide principled machinery for high-order relation modeling but are rarely grounded in multimodal brain tumor MRI classification. This project targets the intersection by constructing a two-stage, mask-aware ROI hypergraph that first aggregates available modalities within anatomical ROIs and then performs prior-guided plus patient-adaptive reasoning across ROI nodes for HGG/LGG classification.

## Claims To Avoid

- Avoid claiming to be the first hypergraph method for missing-modality brain tumor MRI, because `MM-2025-REHYDIL` exists in the matrix.
- Avoid claiming superiority to segmentation methods unless classification experiments support it.
- Avoid claiming modality reconstruction if the model only zeroes or masks missing inputs.
- Avoid treating 2025-2026 preprints as settled baselines without final venue verification.

## Claims That Are Defensible From The Matrix

- Missing-modality multimodal brain tumor MRI is an active research area from 2021 to 2026.
- Most close works target segmentation, not HGG/LGG classification.
- Explicit hypergraph modeling is uncommon in the missing-modality brain tumor MRI subset.
- The closest hypergraph-overlap paper uses a different setting: cross-patient hypergraph segmentation with domain-incremental learning.
- A two-stage within-case ROI hypergraph gives a distinct research angle: anatomical interpretability plus missing-modality-aware classification.

