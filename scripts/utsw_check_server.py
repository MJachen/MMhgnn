from __future__ import annotations

import argparse
import importlib
import json
import platform
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-fast UTSW Linux GPU server environment check.")
    parser.add_argument("--config", default="configs/utsw_idh/server_smoke.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--allow-cpu", action="store_true", help="Development only; production server checks require CUDA.")
    return parser.parse_args()


def _module_version(import_name: str) -> str:
    module = importlib.import_module(import_name)
    return str(getattr(module, "__version__", "installed"))


def main() -> None:
    args = parse_args()
    report = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "dependencies": {},
    }
    for import_name, display_name in [
        ("torch", "PyTorch"),
        ("nibabel", "nibabel"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("sklearn", "sklearn"),
        ("yaml", "yaml"),
    ]:
        report["dependencies"][display_name] = _module_version(import_name)

    import torch
    from datasets.utsw_dataset import UTSWClassificationDataset, load_utsw_records, split_utsw_records
    from utils import load_config

    importlib.import_module("models")
    importlib.import_module("utils.training")
    report["project_imports"] = "pass"
    report["torch_cuda_version"] = torch.version.cuda
    report["cuda_available"] = bool(torch.cuda.is_available())
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available; server smoke requires a CUDA GPU.")
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"UTSW data root does not exist: {data_root}")
    config = load_config(args.config, overrides={"data": {"root": str(data_root)}})
    manifest_path = Path(config["data"]["manifest_csv"])
    split_path = Path(config["data"]["split_json"])
    fingerprint_path = Path(config["data"]["manifest_fingerprint_json"])
    for path in [manifest_path, split_path, fingerprint_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required server artifact is missing: {path}")

    records = load_utsw_records(manifest_path, data_root, fingerprint_path)
    splits = split_utsw_records(
        records,
        split_path,
        require_full_cohort=not bool(config["data"].get("allow_subset_split", False)),
    )
    all_split_records = splits["train"] + splits["val"] + splits["test"]
    rng = np.random.default_rng(int(config["seed"]))
    record = all_split_records[int(rng.integers(0, len(all_split_records)))]
    dataset = UTSWClassificationDataset(
        [record],
        combo_mode="fixed_combo",
        explicit_combo=config["data"]["modalities"],
        fixed_combo=config["data"]["modalities"],
        all_modalities=config["data"]["modalities"],
        target_shape=None,
        crop_mode="full",
        bbox_margin=int(config["data"].get("bbox_margin", 6)),
        random_seed=int(config["seed"]),
        q_core=float(config["model"].get("q_core", 0.4)),
        peri_inner_radius=int(config["model"].get("peri_inner_radius", 3)),
        peri_outer_radius=int(config["model"].get("peri_outer_radius", 7)),
        allow_full_resolution_input=True,
        tumor_mask_source="seg",
    )
    item = dataset[0]
    report["data_root"] = str(data_root)
    report["manifest"] = str(manifest_path)
    report["split"] = str(split_path)
    report["sample_case"] = {
        "case_id": item["case_id"],
        "label": int(item["label"]),
        "images_shape": list(item["images"].shape),
        "roi_masks_shape": list(item["roi_masks"].shape),
        "roi_valid": item["roi_valid"].tolist(),
        "available_modalities": item["available_modalities"].tolist(),
    }
    if list(item["images"].shape[:1]) != [4] or list(item["roi_masks"].shape[:1]) != [5]:
        raise RuntimeError("UTSW sample tensor contract is invalid.")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
