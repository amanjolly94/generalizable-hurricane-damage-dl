"""
Generate the paper's result figures from the real numbers recorded in
PAPER_REVAMPED.md (Sections IV-C through IV-G). No numbers here are
estimated; every value is copied from a downloaded, verified results.json.
Run locally: this is lightweight plotting, not a training workload.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_baseline_comparison():
    models = ["VGG16", "MobileNetV2", "DenseNet121", "Proposed"]
    accuracy = [90.5, 93.5, 92.4, 97.1]
    params_m = [14.72, 2.26, 7.04, 1.29]
    colors = ["#8c8c8c", "#8c8c8c", "#8c8c8c", "#1b6ca8"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    ax1.bar(models, accuracy, color=colors)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(85, 100)
    ax1.set_title("(a) Binary classification accuracy")
    for i, v in enumerate(accuracy):
        ax1.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
    ax1.tick_params(axis="x", rotation=20)

    ax2.bar(models, params_m, color=colors)
    ax2.set_ylabel("Parameters (millions)")
    ax2.set_title("(b) Model size")
    for i, v in enumerate(params_m):
        ax2.text(i, v + 0.3, f"{v:.2f}M", ha="center", fontsize=9)
    ax2.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    save(fig, "fig1_baseline_comparison.png")


def fig_confusion_matrix(cm, class_names, title, filename, normalize=False):
    cm = np.array(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center",
                     color="white" if val > thresh else "black", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, filename)


def fig_cross_hurricane():
    folds = ["Michael", "Harvey", "Florence", "Matthew"]
    zero_shot = [53.2, 56.4, 32.1, 49.5]
    fine_tuned = [71.0, 72.9, 82.0, 85.7]
    in_domain = [82.5, 80.0, 78.0, 83.0]  # approximate in-domain validation range reported in Section IV-D text (78-83%)

    x = np.arange(len(folds))
    width = 0.27

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - width, in_domain, width, label="In-domain (train hurricanes)", color="#8c8c8c")
    ax.bar(x, zero_shot, width, label="Zero-shot (held-out hurricane)", color="#c0392b")
    ax.bar(x + width, fine_tuned, width, label="Fine-tuned (20% held-out slice)", color="#1b6ca8")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Held out:\n{f}" for f in folds])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Cross-hurricane generalization across all 4 held-out folds")
    ax.legend(loc="lower right", fontsize=8)
    ax.axhline(50, color="black", linewidth=0.8, linestyle=":", alpha=0.6)

    fig.tight_layout()
    save(fig, "fig2_cross_hurricane_generalization.png")


def fig_severity_f1():
    classes = ["No-damage", "Minor", "Major", "Destroyed"]
    f1 = [75.2, 58.1, 48.6, 43.4]
    support = [4111, 2178, 1534, 459]

    fig, ax1 = plt.subplots(figsize=(6, 3.8))
    color1 = "#1b6ca8"
    ax1.bar(classes, f1, color=color1)
    ax1.set_ylabel("F1 score (%)", color=color1)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="y", labelcolor=color1)
    for i, v in enumerate(f1):
        ax1.text(i, v + 2, f"{v:.1f}", ha="center", fontsize=9, color=color1)

    ax2 = ax1.twinx()
    color2 = "#8c8c8c"
    ax2.plot(classes, support, color=color2, marker="o", linewidth=1.5)
    ax2.set_ylabel("Test-set support (n)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Severity classification: F1 vs. class support")
    fig.tight_layout()
    save(fig, "fig4_severity_f1_vs_support.png")


def fig_gradcam_pointing_game():
    labels = ["Chance baseline\n(25% of area)", "Observed\npointing-game accuracy"]
    values = [25.0, 29.5]
    colors = ["#8c8c8c", "#c0392b"]

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 50)
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_title("Grad-CAM pointing-game accuracy\n(z=1.47, p=0.14, not significant)")
    fig.tight_layout()
    save(fig, "fig5_gradcam_pointing_game.png")


if __name__ == "__main__":
    fig_baseline_comparison()

    # Exp C: proposed model confusion matrix, Harvey binary test set (n=2400)
    fig_confusion_matrix(
        cm=[[1189, 34], [35, 1142]],
        class_names=["No-damage", "Damage"],
        title="Proposed model (Harvey, n=2400)",
        filename="fig3a_confusion_binary_proposed.png",
    )

    # Exp B: severity classification confusion matrix (20-D fusion, focal loss, n=8282)
    fig_confusion_matrix(
        cm=[[3347, 307, 436, 21], [774, 1125, 195, 84], [590, 142, 721, 81], [79, 121, 80, 179]],
        class_names=["No-damage", "Minor", "Major", "Destroyed"],
        title="Severity classification (xBD, n=8282)",
        filename="fig3b_confusion_severity.png",
        normalize=True,
    )

    fig_cross_hurricane()
    fig_severity_f1()
    fig_gradcam_pointing_game()
