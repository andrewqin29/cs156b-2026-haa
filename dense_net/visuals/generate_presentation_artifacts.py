from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BASELINE_RUN_DIR = Path(
    "/resnick/groups/CS156b/from_central/2026/haa/dense_net_runs/densenet121_run1"
)

BG = "#f8f7f2"
TEXT = "#1f1f1f"
MUTED = "#666666"
GRID = "#d8d5cb"
TRAIN = "#3a86ff"
VAL = "#2a9d8f"
ACCENT = "#f4a261"
BAR = "#577590"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=BASELINE_RUN_DIR)
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix for generated artifacts. Defaults to 'baseline' for the baseline run, otherwise the run directory name.",
    )
    return parser.parse_args()


def _artifact_prefix(run_dir: Path, requested_prefix: str | None) -> str:
    if requested_prefix:
        return requested_prefix
    if run_dir.resolve() == BASELINE_RUN_DIR.resolve():
        return "baseline"
    return run_dir.name


def _model_display_name(run_config: dict) -> str:
    model_name = str(run_config.get("model_name", "DenseNet")).lower()
    if model_name == "densenet121":
        return "DenseNet121"
    if model_name == "densenet169":
        return "DenseNet169"
    return str(run_config.get("model_name", "DenseNet"))


def _load_required_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = TEXT,
) -> None:
    x0, y0, x1, y1 = box
    width, height = _text_size(draw, text, font)
    x = x0 + (x1 - x0 - width) / 2
    y = y0 + (y1 - y0 - height) / 2
    draw.text((x, y), text, font=font, fill=fill)


def _chart_area(draw: ImageDraw.ImageDraw, image: Image.Image, title: str) -> tuple[int, int, int, int]:
    title_font = _font(34, bold=True)
    subtitle_font = _font(18)
    draw.text((60, 40), title, font=title_font, fill=TEXT)
    draw.text((60, 88), "Generated from DenseNet run artifacts", font=subtitle_font, fill=MUTED)
    return 120, 180, image.width - 80, image.height - 120


def _draw_axes(
    draw: ImageDraw.ImageDraw, chart: tuple[int, int, int, int], y_ticks: int = 5
) -> tuple[int, int, int, int]:
    left, top, right, bottom = chart
    draw.line((left, top, left, bottom), fill=TEXT, width=3)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=3)
    for index in range(y_ticks + 1):
        y = top + (bottom - top) * index / y_ticks
        draw.line((left, y, right, y), fill=GRID, width=1)
    return chart


def _draw_auc_curve(metrics: pd.DataFrame, best_epoch: int, output_path: Path, title: str) -> None:
    image = Image.new("RGB", (1400, 900), BG)
    draw = ImageDraw.Draw(image)
    chart = _chart_area(draw, image, title)
    left, top, right, bottom = _draw_axes(draw, chart)

    epochs = metrics["epoch"].tolist()
    train_values = metrics["train_auc"].tolist()
    val_values = metrics["val_auc"].tolist()

    y_min = min(train_values + val_values)
    y_max = max(train_values + val_values)
    pad = max((y_max - y_min) * 0.12, 0.02)
    y_min = max(0.0, y_min - pad)
    y_max = min(1.0, y_max + pad)

    axis_font = _font(18)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = y_max - frac * (y_max - y_min)
        y = top + frac * (bottom - top)
        label = f"{value:.2f}"
        w, h = _text_size(draw, label, axis_font)
        draw.text((left - w - 18, y - h / 2), label, font=axis_font, fill=MUTED)

    def x_pos(epoch: int) -> float:
        if len(epochs) == 1:
            return (left + right) / 2
        return left + (epoch - min(epochs)) * (right - left) / (max(epochs) - min(epochs))

    def y_pos(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / max(y_max - y_min, 1e-9)

    for epoch in epochs:
        x = x_pos(epoch)
        label = str(epoch)
        w, _ = _text_size(draw, label, axis_font)
        draw.text((x - w / 2, bottom + 18), label, font=axis_font, fill=MUTED)

    best_x = x_pos(best_epoch)
    draw.line((best_x, top, best_x, bottom), fill=ACCENT, width=2)

    def draw_series(values: list[float], color: str) -> None:
        points = [(x_pos(epoch), y_pos(value)) for epoch, value in zip(epochs, values)]
        draw.line(points, fill=color, width=5, joint="curve")
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=color)

    draw_series(train_values, TRAIN)
    draw_series(val_values, VAL)

    legend_font = _font(22)
    legend_y = 120
    draw.rounded_rectangle((1030, legend_y, 1320, legend_y + 120), radius=20, fill="#ffffff", outline=GRID)
    draw.rectangle((1060, legend_y + 26, 1095, legend_y + 36), fill=TRAIN)
    draw.text((1110, legend_y + 16), "Train AUC", font=legend_font, fill=TEXT)
    draw.rectangle((1060, legend_y + 66, 1095, legend_y + 76), fill=VAL)
    draw.text((1110, legend_y + 56), "Val AUC", font=legend_font, fill=TEXT)
    draw.text((1060, legend_y + 92), f"Best epoch: {best_epoch}", font=_font(18), fill=ACCENT)

    image.save(output_path)


def _draw_summary_table(
    run_config: dict, best_metrics: dict, metrics: pd.DataFrame, output_path: Path, title: str
) -> None:
    image = Image.new("RGB", (1200, 720), BG)
    draw = ImageDraw.Draw(image)
    _chart_area(draw, image, title)

    rows = [
        ("Model", run_config.get("model_name", "unknown")),
        ("Image size", str(run_config.get("image_size", "unknown"))),
        ("Best val AUC", f"{best_metrics['best_val_auc']:.4f}"),
        ("Best epoch", str(best_metrics["best_epoch"])),
        ("Epochs run", str(len(metrics))),
        ("Learning rate", f"{run_config.get('lr', 'unknown')}"),
        ("Batch size", str(run_config.get("batch_size", "unknown"))),
        ("Warmup epochs", str(run_config.get("warmup_epochs", "unknown"))),
    ]
    if "patience" in run_config:
        rows.append(("Patience", str(run_config["patience"])))

    x0, y0, x1, y1 = 140, 180, 1060, 600
    draw.rounded_rectangle((x0, y0, x1, y1), radius=24, fill="#ffffff", outline=GRID, width=2)

    title_font = _font(26, bold=True)
    body_font = _font(24)
    row_height = (y1 - y0 - 50) / len(rows)
    split_x = x0 + 360

    draw.text((x0 + 40, y0 + 16), "Metric", font=title_font, fill=TEXT)
    draw.text((split_x + 30, y0 + 16), "Value", font=title_font, fill=TEXT)
    draw.line((split_x, y0 + 10, split_x, y1 - 15), fill=GRID, width=2)

    for idx, (label, value) in enumerate(rows):
        row_top = y0 + 50 + idx * row_height
        if idx > 0:
            draw.line((x0 + 20, row_top, x1 - 20, row_top), fill=GRID, width=1)
        draw.text((x0 + 40, row_top + 14), label, font=body_font, fill=TEXT)
        draw.text((split_x + 30, row_top + 14), value, font=body_font, fill=TEXT)

    image.save(output_path)


def _draw_per_label_auc(best_metrics: dict, output_path: Path, title: str) -> None:
    label_aucs = best_metrics["val_per_label_auc"]
    ordered = sorted(label_aucs.items(), key=lambda item: item[1], reverse=True)

    image = Image.new("RGB", (1600, 900), BG)
    draw = ImageDraw.Draw(image)
    chart = _chart_area(draw, image, title)
    left, top, right, bottom = _draw_axes(draw, chart)

    axis_font = _font(18)
    bar_font = _font(18)
    n = len(ordered)
    bar_gap = 20
    total_width = right - left
    bar_width = (total_width - (n + 1) * bar_gap) / n

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = 1.0 - frac
        y = top + frac * (bottom - top)
        label = f"{value:.2f}"
        w, h = _text_size(draw, label, axis_font)
        draw.text((left - w - 18, y - h / 2), label, font=axis_font, fill=MUTED)

    for idx, (label, auc) in enumerate(ordered):
        x = left + bar_gap + idx * (bar_width + bar_gap)
        bar_top = bottom - auc * (bottom - top)
        draw.rounded_rectangle((x, bar_top, x + bar_width, bottom), radius=10, fill=BAR)
        auc_label = f"{auc:.3f}"
        w, h = _text_size(draw, auc_label, axis_font)
        draw.text((x + (bar_width - w) / 2, bar_top - h - 10), auc_label, font=axis_font, fill=TEXT)

        wrapped = label.replace(" ", "\n")
        _draw_centered_text(
            draw,
            (int(x - 10), bottom + 24, int(x + bar_width + 10), bottom + 110),
            wrapped,
            bar_font,
            fill=TEXT,
        )

    image.save(output_path)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    metrics = _load_required_csv(run_dir / "metrics.csv")
    best_metrics = _load_required_json(run_dir / "best_metrics.json")
    run_config = _load_required_json(run_dir / "run_config.json")
    artifact_prefix = _artifact_prefix(run_dir, args.prefix)
    model_name = _model_display_name(run_config)

    presentation_dir = run_dir / "presentation"
    presentation_dir.mkdir(parents=True, exist_ok=True)

    _draw_auc_curve(
        metrics,
        int(best_metrics["best_epoch"]),
        presentation_dir / f"{artifact_prefix}_auc_curve.png",
        f"{model_name} AUC Curves",
    )
    _draw_summary_table(
        run_config,
        best_metrics,
        metrics,
        presentation_dir / f"{artifact_prefix}_summary_table.png",
        "Run Summary",
    )
    _draw_per_label_auc(
        best_metrics,
        presentation_dir / f"{artifact_prefix}_per_label_auc.png",
        "Validation AUC by Label",
    )

    print(f"Generated presentation artifacts in {presentation_dir}")


if __name__ == "__main__":
    main()
