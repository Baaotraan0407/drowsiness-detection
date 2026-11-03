"""Draw the validation confusion matrices from results/metrics.json
into results/confusion_matrices.png (no re-training needed).

Usage:  python plot_confusion.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

m = json.load(open("results/metrics.json"))
titles = {"eye": "Eye CNN (subject-independent split)", "mouth": "Mouth CNN (random split)"}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, k in zip(axs, ["eye", "mouth"]):
    cm = np.array(m[k]["confusion_matrix"]["rows_true_cols_pred"])
    labels = m[k]["confusion_matrix"]["labels"]
    rown = cm / cm.sum(1, keepdims=True)
    ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({rown[i, j]:.1%})", ha="center", va="center",
                    fontsize=12, color="white" if rown[i, j] > 0.5 else "#1a1a1a")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{titles[k]}\nn = {m[k]['n_val']:,}, accuracy = {m[k]['accuracy']:.1%}", fontsize=11)
fig.suptitle("Validation confusion matrices (rows sum to 100%)", fontsize=12)
fig.tight_layout()
fig.savefig("results/confusion_matrices.png", dpi=150)
print("Saved results/confusion_matrices.png")
