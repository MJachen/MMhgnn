from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_SEEDS = [42, 123, 3407]


def parse_args():
    parser = argparse.ArgumentParser(description="Run hyperedge ablation experiments")
    parser.add_argument("--config_root", type=str, default="configs/ablation")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--output_root", type=str, default="outputs/ablation")
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_root = Path(args.config_root)
    config_paths = sorted(config_root.glob("*.yaml"))
    if args.configs:
        wanted = set(args.configs)
        config_paths = [p for p in config_paths if p.stem in wanted or p.name in wanted]

    if not config_paths:
        raise FileNotFoundError(f"No yaml configs found under {config_root}")

    logs_dir = Path(args.output_root) / "_run_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary = {"success": [], "failed": []}
    print(f"Discovered {len(config_paths)} configs under {config_root}")

    for config_path in config_paths:
        exp_name = config_path.stem
        for seed in args.seeds:
            output_dir = Path(args.output_root) / exp_name / f"seed_{seed}"
            ckpt_path = output_dir / "checkpoints" / "best.pt"
            if args.skip_existing and ckpt_path.exists():
                print(f"Skipping existing run: {exp_name} seed={seed}")
                summary["success"].append({"experiment": exp_name, "seed": seed, "status": "skipped"})
                continue

            cmd = [
                args.python,
                "train.py",
                "--config",
                str(config_path),
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
            ]
            log_path = logs_dir / f"{exp_name}_seed_{seed}.log"
            print("Running:", " ".join(cmd))
            with open(log_path, "w", encoding="utf-8") as log_file:
                result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

            if result.returncode == 0:
                print(f"Finished: {exp_name} seed={seed}")
                summary["success"].append({"experiment": exp_name, "seed": seed, "log": str(log_path)})
            else:
                print(f"Failed: {exp_name} seed={seed}. See {log_path}")
                summary["failed"].append({"experiment": exp_name, "seed": seed, "log": str(log_path), "returncode": result.returncode})
                if args.stop_on_error:
                    with open(Path(args.output_root) / "run_ablation_summary.json", "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                    raise SystemExit(result.returncode)

    with open(Path(args.output_root) / "run_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Done. Summary written to", Path(args.output_root) / "run_ablation_summary.json")


if __name__ == "__main__":
    main()
