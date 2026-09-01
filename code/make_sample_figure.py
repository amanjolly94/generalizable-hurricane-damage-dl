"""
Dataset sample figure, mirroring the original submission's Fig. 1 (damaged
vs. non-damaged training samples) and Fig. 2 (RGB colorspace visualization).
Uses two real images downloaded from the Harvey dataset (Kaggle:
kmader/satellite-images-of-hurricane-damage), not synthetic placeholders.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ieee_submission", "samples")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

DAMAGE_IMG = os.path.join(SAMPLES_DIR, "-93.548123_30.900623.jpeg")
NO_DAMAGE_IMG = os.path.join(SAMPLES_DIR, "-95.061894_30.007746.jpeg")


def fig_dataset_samples():
    damage = np.array(Image.open(DAMAGE_IMG).convert("RGB"))
    no_damage = np.array(Image.open(NO_DAMAGE_IMG).convert("RGB"))

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    axes[0].imshow(no_damage)
    axes[0].set_title("(a) No-damage")
    axes[0].axis("off")
    axes[1].imshow(damage)
    axes[1].set_title("(b) Damage")
    axes[1].axis("off")
    fig.suptitle("Hurricane Harvey dataset samples", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig0a_dataset_samples.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig0a_dataset_samples.png")


def fig_colorspace():
    damage = np.array(Image.open(DAMAGE_IMG).convert("RGB"))
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.6))
    axes[0].imshow(damage)
    axes[0].set_title("RGB")
    for i, (name, cmap) in enumerate(zip(["R", "G", "B"], ["Reds", "Greens", "Blues"])):
        axes[i + 1].imshow(damage[:, :, i], cmap=cmap)
        axes[i + 1].set_title(name)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("RGB channel decomposition (damage sample)", y=1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig0b_colorspace.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig0b_colorspace.png")


if __name__ == "__main__":
    fig_dataset_samples()
    fig_colorspace()
