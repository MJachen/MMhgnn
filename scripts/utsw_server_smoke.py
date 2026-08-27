from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.optim import AdamW


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import HybridHypergraphClassifier
from utils import load_config, set_seed
from utils.runner import move_batch_to_device
from utils.training import build_dataloader, build_datasets


PATTERNS = {
    "full": ["t2", "t1ce", "t1", "flair"],
    "missing_t1ce": ["t2", "t1", "flair"],
    "single_flair": ["flair"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real UTSW forward/backward/optimizer server smoke path.")
    parser.add_argument("--config", default="configs/utsw_idh/server_smoke.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _assert_shape(tensor: torch.Tensor, expected_prefix: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape[: len(expected_prefix)]) != expected_prefix:
        raise RuntimeError(f"{name} shape mismatch: expected prefix {expected_prefix}, found {tuple(tensor.shape)}")


def main() -> None:
    args = parse_args()
    overrides = {"data": {"root": str(Path(args.data_root).expanduser().resolve())}}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    config = load_config(args.config, overrides=overrides)
    set_seed(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("UTSW server smoke requires CUDA.")
    device = torch.device("cuda")
    model = HybridHypergraphClassifier(config).to(device)
    if float(model.get_classifier_info()["no_t1ce_t1_penalty"]) != 0.0:
        raise RuntimeError("UTSW server smoke requires no_t1ce_t1_penalty=0.")
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=float(config["train"]["lr"]), weight_decay=float(config["train"].get("weight_decay", 0.0)))
    results = {}
    data_order = list(config["data"]["modalities"])
    gate_order = list(model.get_classifier_info()["mask_order"])

    for name, combo in PATTERNS.items():
        _, _, dataset, _ = build_datasets(config, explicit_eval_combo=combo)
        loader = build_dataloader(dataset, batch_size=1, num_workers=0, shuffle=False)
        batch = move_batch_to_device(next(iter(loader)), device)
        _assert_shape(batch["images"], (1, 4), "images")
        _assert_shape(batch["roi_masks"], (1, 5), "roi_masks")
        _assert_shape(batch["roi_valid"], (1, 5), "roi_valid")
        _assert_shape(batch["available_modalities"], (1, 4), "available_modalities")
        for index, modality in enumerate(data_order):
            expected = 1.0 if modality in combo else 0.0
            if float(batch["available_modalities"][0, index]) != expected:
                raise RuntimeError(f"{name}: availability mask is wrong for {modality}")
            if expected == 0.0 and torch.count_nonzero(batch["images"][0, index]).item() != 0:
                raise RuntimeError(f"{name}: missing modality {modality} was not zeroed")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        _assert_shape(output["logits"], (1, 2), "logits")
        if not torch.isfinite(output["logits"]).all() or not torch.isfinite(output["prob"]).all():
            raise RuntimeError(f"{name}: logits/prob contains NaN or Inf")
        for gate_index, modality in enumerate(gate_order):
            if modality not in combo and not torch.all(output["modality_gates"][..., gate_index] == 0):
                raise RuntimeError(f"{name}: missing modality gate is non-zero for {modality}")
        loss = criterion(output["logits"], batch["label"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: loss contains NaN or Inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"].get("grad_clip", 0.0)))
        optimizer.step()
        results[name] = {
            "combo": combo,
            "case_id": batch["case_id"][0],
            "loss": float(loss.detach().cpu()),
            "images_shape": list(batch["images"].shape),
            "roi_masks_shape": list(batch["roi_masks"].shape),
            "roi_valid_shape": list(batch["roi_valid"].shape),
            "available_modalities": batch["available_modalities"].detach().cpu().tolist(),
            "logits_shape": list(output["logits"].shape),
            "gating_zero_assertion": "pass",
        }
        del batch, output, loss
        torch.cuda.empty_cache()

    report = {"status": "pass", "device": torch.cuda.get_device_name(0), "patterns": results}
    if args.output_dir:
        output_path = Path(args.output_dir) / "server_smoke_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
