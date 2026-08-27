from __future__ import annotations

import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Iterable, Optional, Tuple

import pandas as pd


UTSW_MODALITIES = ["t2", "t1ce", "t1", "flair"]
UTSW_IDH_WILDTYPE = 0
UTSW_IDH_MUTANT = 1

UTSW_FINGERPRINT_COLUMNS = [
    "case_id",
    "patient_id",
    "normalized_idh_label",
    "t2_path",
    "t1ce_path",
    "t1_path",
    "flair_path",
    "segmentation_path",
    "mri_source_version",
    "segmentation_source",
    "geometry_qc_status",
    "eligible",
    "exclusion_reason",
]


def validate_relative_data_path(value: object) -> PurePosixPath:
    """Validate one platform-neutral path stored relative to ``data.root``."""
    text = str(value).strip()
    if not text:
        raise ValueError("UTSW manifest contains an empty data path.")
    if "\\" in text:
        raise ValueError(f"UTSW manifest paths must use POSIX separators: {text}")
    path = PurePosixPath(text)
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise ValueError(f"UTSW manifest path must be relative to data.root: {text}")
    return path


def resolve_data_path(data_root: str | Path, relative_path: object) -> Path:
    relative = validate_relative_data_path(relative_path)
    return Path(data_root).expanduser().resolve().joinpath(*relative.parts)


def canonical_manifest_fingerprint(manifest: pd.DataFrame) -> str:
    """Hash cohort identity and portable file mappings, independent of CSV formatting."""
    missing = sorted(set(UTSW_FINGERPRINT_COLUMNS).difference(manifest.columns))
    if missing:
        raise ValueError(f"Cannot fingerprint UTSW manifest; missing columns: {missing}")
    canonical = manifest.loc[:, UTSW_FINGERPRINT_COLUMNS].copy()
    for column in canonical.columns:
        canonical[column] = canonical[column].astype(str).str.strip()
    canonical = canonical.sort_values(["case_id", "patient_id"], kind="stable")
    records = canonical.to_dict(orient="records")
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_idh_label(value: object) -> Tuple[Optional[int], str]:
    """Normalize the observed UTSW IDH vocabulary without guessing unknowns."""
    text = "" if value is None else str(value).strip()
    normalized = text.casefold().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    if normalized in {"wild type", "wildtype", "idh wild type", "idh wildtype"}:
        return UTSW_IDH_WILDTYPE, ""
    if normalized in {"mutated", "mutant", "idh mutated", "idh mutant"}:
        return UTSW_IDH_MUTANT, ""
    if normalized in {"", "na", "n/a", "unknown", "not tested", "indeterminate"}:
        return None, f"excluded_idh_{normalized.replace(' ', '_') or 'empty'}"
    return None, f"excluded_idh_ambiguous:{text}"


def load_utsw_metadata(path: str | Path) -> pd.DataFrame:
    """Load UTSW metadata and append explicit normalized IDH fields."""
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"Subject ID", "IDH"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"UTSW metadata is missing required columns: {missing}")

    df = df.copy()
    df["Subject ID"] = df["Subject ID"].astype(str).str.strip()
    if (df["Subject ID"] == "").any():
        raise ValueError("UTSW metadata contains empty Subject ID values.")
    duplicates = df.loc[df["Subject ID"].duplicated(keep=False), "Subject ID"].tolist()
    if duplicates:
        raise ValueError(f"UTSW metadata contains duplicate Subject ID values: {duplicates[:10]}")

    parsed = df["IDH"].map(normalize_idh_label)
    df["original_idh_label"] = df["IDH"]
    df["normalized_idh_label"] = parsed.map(lambda item: item[0])
    df["label_exclusion_reason"] = parsed.map(lambda item: item[1])
    return df
