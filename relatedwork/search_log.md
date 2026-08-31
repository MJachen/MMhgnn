# Search Log

Date: 2026-07-06  
Scope requested: complete `literature_matrix.csv` and `search_log.md` first; do not draft Related Work.  
Topic window: 2021-2026 papers on hypergraph learning, brain tumor MRI classification/segmentation, and missing-modality multimodal MRI.

## Workflow

- Used the `nature-academic-search` workflow category: `multi-source-search`.
- Academic-search MCP tools were not exposed in the available tool list, so live web/arXiv search was used as the verification path.
- Inclusion required a current source page with title, authors, year/date, and topic match.
- Venue claims were kept conservative. When only arXiv metadata was verified, `venue_or_status` is recorded as `arXiv preprint` or `arXiv preprint / challenge report`.
- No Related Work prose was written.

## Search Queries

| Query | Purpose | Outcome |
|---|---|---|
| `2021 2026 hypergraph learning brain tumor MRI classification segmentation missing modality multimodal MRI paper DOI` | broad cross-topic seed search | surfaced missing-modality and BraTS papers; too broad for final filtering |
| `missing modality multimodal brain tumor segmentation MRI 2021 2026 DOI` | missing-modality MRI focus | found M3AE and related missing-modality works |
| `hypergraph neural network brain tumor MRI classification segmentation 2021 DOI` | direct hypergraph + tumor MRI check | no strong direct peer-reviewed hits surfaced |
| `BraTS missing modalities multimodal MRI segmentation 2022 2023 2024 paper DOI` | BraTS missing-modality focus | found mmFormer, M3AE, and challenge-context papers |
| `ACN: Adversarial Co-training Network for Brain Tumor Segmentation with Missing Modalities` | verify ACN metadata | included ACN as 2021 missing-modality baseline |
| `M3AE Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities DOI` | verify M3AE metadata | included M3AE as 2023 missing-modality representation-learning work |
| `mmFormer brain tumor segmentation missing modalities DOI` | verify mmFormer metadata | included mmFormer as 2022 incomplete multimodal transformer baseline |
| `TransBTS Multimodal Brain Tumor Segmentation Using Transformer DOI` | verify segmentation baseline | included TransBTS as 2021 multimodal segmentation baseline |
| `Swin UNETR brain tumors MRI DOI` | verify transformer segmentation baseline | included Swin UNETR as 2022 BraTS transformer segmentation baseline |
| `RFNet incomplete multi-modal brain tumor segmentation arxiv` | search for incomplete multimodal segmentation variants | surfaced DIGEST, MAF-Net, mmFormer, and semi-paired/curriculum papers |
| `missing modality brain tumor segmentation 2024 arXiv` | recent missing-modality papers | included MPPA/divergence and alignment-based 2024 preprints |
| `brain tumor segmentation with missing modalities 2025` | 2025 missing-modality papers | included AdaMM and ReHyDIL candidates |
| `Set-Inclusive Uncertainty Modeling for Robust Brain Tumor Segmentation` | 2026 missing-modality uncertainty check | included SIUM as a 2026 preprint within the date window |
| `UniGNN a Unified Framework for Graph and Hypergraph Neural Networks` | hypergraph learning verification | included UniGNN |
| `You are AllSet A Multiset Function Framework for Hypergraph Neural Networks` | hypergraph learning verification | included AllSet |
| `Equivariant Hypergraph Diffusion Neural Operators` | hypergraph learning verification | included ED-HNN |
| `Hypergraph Transformer for Semi-Supervised Classification` | hypergraph transformer verification | included HyperGT |
| `Recent Advances in Hypergraph Neural Networks` | recent HGNN survey/taxonomy | included as review/taxonomy candidate |
| `HGG LGG MRI classification deep learning 2021 BraTS` | classification-side search | included explanation-guided glioma classification as a classification-context paper |

## Inclusion Rules Applied

- Year must be 2021-2026.
- Topic must fit at least one of:
  - hypergraph learning / HGNN / hypergraph transformer / hypergraph review;
  - brain tumor MRI segmentation or glioma MRI classification;
  - missing-modality or incomplete multimodal brain MRI segmentation.
- Source must provide verifiable metadata in the current search session.
- Direct project relevance was prioritized:
  - highest: hypergraph + missing-modality + brain tumor MRI;
  - high: missing-modality multimodal BraTS segmentation;
  - medium: modern BraTS multimodal segmentation baselines;
  - background: general hypergraph-learning methods.

## Exclusion Notes

- Pre-2021 missing-modality classics were not included because the user requested 2021-2026. Example excluded: `Brain Tumor Segmentation on MRI with Missing Modalities` from 2019.
- Papers with search-result snippets but no clear metadata match or weak topic fit were not added.
- General medical image segmentation papers were excluded unless the search result explicitly tied them to brain tumor MRI/BraTS or missing modalities.
- Final peer-reviewed venue/DOI was not asserted when the current search only verified an arXiv page.

## Current Matrix Summary

- Total entries: 21.
- Hypergraph learning background: 5.
- Brain tumor MRI segmentation/classification context: 6.
- Missing-modality multimodal MRI: 10, counting the hybrid hypergraph/missing-modality entry.
- Direct hypergraph + missing-modality + brain tumor MRI overlap: 1 high-priority recent candidate (`MM-2025-REHYDIL`).

## Follow-up Checks Before Writing Related Work

- Verify final publication venues and DOIs for high-priority arXiv entries, especially ACN, mmFormer, M3AE, DIGEST, ReHyDIL, and AdaMM.
- Decide whether preprint-only 2025-2026 works should be cited in the main paper or kept for discussion/future-work context.
- Add BibTeX keys only after the `.bib` target file is known.
