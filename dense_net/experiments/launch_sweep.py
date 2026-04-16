from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path("/resnick/groups/CS156b/from_central/2026/haa/asqin/cs156b-2026-haa")
TRAIN_SLURM = REPO_ROOT / "dense_net" / "train" / "train_densenet.slurm"
RUNS_ROOT = Path("/resnick/groups/CS156b/from_central/2026/haa/dense_net_runs")
TORCH_HOME = Path("/resnick/groups/CS156b/from_central/2026/haa/asqin/.torch_test_cache")

MANIFESTS_224 = Path(
    "/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests_preprocessed"
)
MANIFESTS_320 = Path(
    "/resnick/groups/CS156b/from_central/2026/haa/dense_net_data/manifests_preprocessed_320"
)


@dataclass(frozen=True)
class SweepRun:
    run_name: str
    model_name: str
    image_size: int
    batch_size: int
    lr: float
    warmup_epochs: int
    patience: int
    epochs: int = 15


SWEEP_RUNS: list[SweepRun] = [
    SweepRun("d121_224_lr5e5", "densenet121", 224, 32, 5e-5, 2, 4),
    SweepRun("d121_224_w3", "densenet121", 224, 32, 1e-4, 3, 4),
    SweepRun("d169_224_lr1e4", "densenet169", 224, 32, 1e-4, 2, 4),
    SweepRun("d169_224_lr5e5", "densenet169", 224, 32, 5e-5, 2, 4),
    SweepRun("d121_320_lr1e4", "densenet121", 320, 16, 1e-4, 2, 4),
    SweepRun("d169_320_lr1e4", "densenet169", 320, 16, 1e-4, 2, 4),
    SweepRun("d169_320_lr5e5", "densenet169", 320, 16, 5e-5, 3, 5),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or submit the first DenseNet sweep."
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit the jobs with sbatch. Default is preview-only.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional subset of run names to preview/submit.",
    )
    return parser.parse_args()


def manifests_for_image_size(image_size: int) -> tuple[Path, Path]:
    manifest_root = MANIFESTS_224 if image_size == 224 else MANIFESTS_320
    return (
        manifest_root / "train_manifest_preprocessed.csv",
        manifest_root / "val_manifest_preprocessed.csv",
    )


def selected_runs(run_names: list[str] | None) -> list[SweepRun]:
    if not run_names:
        return SWEEP_RUNS
    wanted = set(run_names)
    selected = [run for run in SWEEP_RUNS if run.run_name in wanted]
    missing = sorted(wanted - {run.run_name for run in selected})
    if missing:
        raise ValueError(f"Unknown run names: {', '.join(missing)}")
    return selected


def format_preview(runs: list[SweepRun]) -> str:
    headers = [
        "run_name",
        "model",
        "img",
        "bs",
        "lr",
        "warmup",
        "patience",
        "train_manifest",
    ]
    rows = []
    for run in runs:
        train_csv, _ = manifests_for_image_size(run.image_size)
        rows.append(
            [
                run.run_name,
                run.model_name,
                str(run.image_size),
                str(run.batch_size),
                f"{run.lr:g}",
                str(run.warmup_epochs),
                str(run.patience),
                str(train_csv),
            ]
        )
    widths = [
        max(len(header), max(len(row[idx]) for row in rows)) for idx, header in enumerate(headers)
    ]
    header_line = "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    divider = "  ".join("-" * widths[idx] for idx in range(len(headers)))
    body = [
        "  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers)))
        for row in rows
    ]
    return "\n".join([header_line, divider, *body])


def submit_run(run: SweepRun) -> None:
    train_csv, val_csv = manifests_for_image_size(run.image_size)
    out_root = RUNS_ROOT / run.run_name
    env = os.environ.copy()
    env.update(
        {
            "RUN_NAME": run.run_name,
            "MODEL_NAME": run.model_name,
            "IMAGE_SIZE": str(run.image_size),
            "BATCH_SIZE": str(run.batch_size),
            "LR": str(run.lr),
            "WARMUP_EPOCHS": str(run.warmup_epochs),
            "PATIENCE": str(run.patience),
            "EPOCHS": str(run.epochs),
            "TRAIN_CSV": str(train_csv),
            "VAL_CSV": str(val_csv),
            "OUT_ROOT": str(out_root),
            "TORCH_HOME_DIR": str(TORCH_HOME),
        }
    )
    subprocess.run(["sbatch", str(TRAIN_SLURM)], cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    runs = selected_runs(args.runs)
    print(format_preview(runs))
    print()
    if not args.submit:
        print("Preview only. Re-run with --submit to launch these jobs.")
        return

    for run in runs:
        print(f"Submitting {run.run_name}...")
        submit_run(run)


if __name__ == "__main__":
    main()
