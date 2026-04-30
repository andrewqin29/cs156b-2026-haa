"""
Reproduce the buggy Patient Demographics figure from the first presentation.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIN_CSV = Path("/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv")
C_POS, C_NEG = "#4C8CB5", "#E07B54"

df = pd.read_csv(TRAIN_CSV)
frontal = df[df["Frontal/Lateral"] == "Frontal"]

n_male   = len(frontal[frontal["Sex"] != "Unknown"])                       # 152,297
n_female = len(frontal[frontal["Sex"].str.startswith("F", na=False)])      # 63,053

male_sub   = frontal[frontal["Sex"].str.startswith("M", na=False)]
female_sub = frontal[frontal["Sex"].str.startswith("F", na=False)]

bins = np.linspace(frontal["Age"].min(), frontal["Age"].max(), 31)

male_counts,   _ = np.histogram(male_sub["Age"],   bins=bins)
female_counts, _ = np.histogram(female_sub["Age"], bins=bins)

bin_centers = (bins[:-1] + bins[1:]) / 2
width = bins[1] - bins[0]

fig, ax = plt.subplots(figsize=(10, 6))

ax.set_facecolor("#EBEBEB")
fig.patch.set_facecolor("white")

ax.yaxis.grid(True, color="white", linewidth=0.8)
ax.set_axisbelow(True)

ax.bar(bin_centers, female_counts, width=width, color=C_NEG, label=f"Female (n={n_female:,})",
       linewidth=0.5, edgecolor="white")
ax.bar(bin_centers, male_counts,   width=width, color=C_POS, label=f"Male (n={n_male:,})",
       linewidth=0.5, edgecolor="white", bottom=female_counts)

ax.set_title("Age and Sex Distribution", fontsize=12, pad=10)
ax.set_xlabel("Age (years)", fontsize=10)
ax.set_ylabel("Number of patients", fontsize=10)
ax.legend(loc="upper left", framealpha=1)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig("demographics_bug_reproduced_stacked.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved → demographics_bug_reproduced_stacked.png")