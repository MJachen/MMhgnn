# UTSW-IDH server-ready runbook

The UTSW manifest stores all MRI and FeTS paths relative to `UTSW_DATA_ROOT`.
Do not copy local `outputs/`, checkpoints, caches, or ROI overlay images as code artifacts.

## Required data

Copy the UTSW-Glioma directory without renaming case directories or files. The expected selected files are:

- `BTxxxx/brain_t2.nii.gz`
- `BTxxxx/brain_t1ce.nii.gz`
- `BTxxxx/brain_t1.nii.gz`
- `BTxxxx/brain_flair.nii.gz`
- `BTxxxx/tumorseg_FeTS.nii.gz`

The frozen cohort contains 618 eligible cases: 442 IDH-WT (`0`) and 176 IDH-mutant (`1`).

## Environment variables

```bash
export UTSW_DATA_ROOT=/absolute/server/path/UTSW-Glioma
export OUTPUT_ROOT=/absolute/server/path/utsw_idh_outputs
export GPU_ID=0
export PYTHON_BIN=python
```

## Required execution order

```bash
bash scripts/server/utsw_server_smoke.sh
bash scripts/server/utsw_tiny_overfit.sh
bash scripts/server/utsw_full_baseline.sh
bash scripts/server/utsw_missing15.sh
```

Stop after any failed command. Do not run `utsw_missing15.sh` until the full-modality baseline is accepted.

Resume tiny/full training with the corresponding `last.pt`:

```bash
export RESUME_CHECKPOINT="${OUTPUT_ROOT}/full_baseline_seed42/checkpoints/last.pt"
bash scripts/server/utsw_full_baseline.sh
```

The missing-15 script defaults to `${OUTPUT_ROOT}/full_baseline_seed42/checkpoints/best.pt`.
Override it only with an accepted full-baseline checkpoint:

```bash
export FULL_BASELINE_CHECKPOINT=/absolute/server/path/to/best.pt
bash scripts/server/utsw_missing15.sh
```
