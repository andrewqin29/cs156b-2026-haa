"""
plot_training_curves.py — generate training curves from metrics.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="first_run.csv")
    p.add_argument("--output",  default="training_curves.png")
    p.add_argument("--warmup_epochs", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.metrics)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ResNet50 Training Curves", fontsize=14, fontweight="medium", y=1.01)

    ax1.plot(df["epoch"], df["train_auc"], label="Train", color="#378ADD", linewidth=2, marker="o", markersize=4)
    ax1.plot(df["epoch"], df["val_auc"],   label="Val",   color="#1D9E75", linewidth=2, marker="o", markersize=4, linestyle="--")
    best_epoch = df.loc[df["val_auc"].idxmax(), "epoch"]
    ax1.axvline(best_epoch, color="#EF9F27", linewidth=1.2, linestyle=":", label=f"Best val AUC (epoch {best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("AUC")
    ax1.set_title("AUC")
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["epoch"], df["train_loss"], label="Train", color="#378ADD", linewidth=2, marker="o", markersize=4)
    ax2.plot(df["epoch"], df["val_loss"],   label="Val",   color="#1D9E75", linewidth=2, marker="o", markersize=4, linestyle="--")
    ax2.axvline(best_epoch, color="#EF9F27", linewidth=1.2, linestyle=":", label=f"Best val AUC (epoch {best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()