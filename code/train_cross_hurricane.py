"""
Exp A: cross-hurricane generalization. Train on 3 of xBD's 4 hurricanes,
evaluate zero-shot on the held-out 4th, then fine-tune on a small labeled
slice of the held-out hurricane and re-evaluate.

Run inside a Kaggle kernel with GPU enabled, code dataset mounted, and the
xbd-hurricane-preprocess kernel's output available via kernel_sources.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_full_model, IMAGE_SIZE
from train import samples_to_arrays, normalize_geo, evaluate_binary

HURRICANES = ("hurricane-harvey", "hurricane-florence", "hurricane-matthew", "hurricane-michael")


def load_xbd_samples_by_disaster(root):
    """Returns {disaster_name: [(path, lat, lon, binary_label, severity_label), ...]}."""
    metadata_path = os.path.join(root, "metadata.csv")
    patches_dir = os.path.join(root, "patches")
    by_disaster = {h: [] for h in HURRICANES}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            disaster = row["disaster"]
            if disaster not in by_disaster:
                continue
            full_path = os.path.join(patches_dir, row["patch_filename"])
            if not os.path.exists(full_path):
                continue
            by_disaster[disaster].append((
                full_path, float(row["lat"]), float(row["lon"]),
                int(row["binary_label"]), int(row["severity_label"]),
            ))
    return by_disaster


def run(root, held_out, epochs, finetune_epochs, finetune_fraction, out_dir, max_samples_per_hurricane=None, seed=42):
    np.random.seed(seed)
    by_disaster = load_xbd_samples_by_disaster(root)
    for h, samples in by_disaster.items():
        print(f"{h}: {len(samples)} samples")

    train_hurricanes = [h for h in HURRICANES if h != held_out]
    train_samples = []
    for h in train_hurricanes:
        s = by_disaster[h]
        if max_samples_per_hurricane and len(s) > max_samples_per_hurricane:
            idx = np.random.choice(len(s), max_samples_per_hurricane, replace=False)
            s = [s[i] for i in idx]
        train_samples.extend(s)

    held_out_samples = by_disaster[held_out]
    if max_samples_per_hurricane and len(held_out_samples) > max_samples_per_hurricane:
        idx = np.random.choice(len(held_out_samples), max_samples_per_hurricane, replace=False)
        held_out_samples = [held_out_samples[i] for i in idx]

    print(f"train hurricanes: {train_hurricanes} ({len(train_samples)} samples)")
    print(f"held-out hurricane: {held_out} ({len(held_out_samples)} samples)")

    train_images, train_geos, train_labels = samples_to_arrays(train_samples, num_classes=1)
    train_geos = normalize_geo(train_geos)

    held_images, held_geos, held_labels = samples_to_arrays(held_out_samples, num_classes=1)
    held_geos = normalize_geo(held_geos)

    model = build_full_model(num_classes=1, use_residual=True)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        [train_images, train_geos], train_labels.astype("float32"),
        validation_split=0.1, epochs=epochs, batch_size=32, verbose=2,
    )

    zero_shot_pred = model.predict([held_images, held_geos], verbose=0)
    zero_shot_metrics = evaluate_binary(held_labels, zero_shot_pred)
    print(f"zero-shot on {held_out}: {zero_shot_metrics}")

    n_held = len(held_images)
    perm = np.random.permutation(n_held)
    ft_split = int(n_held * finetune_fraction)
    ft_idx, eval_idx = perm[:ft_split], perm[ft_split:]

    model.fit(
        [held_images[ft_idx], held_geos[ft_idx]], held_labels[ft_idx].astype("float32"),
        epochs=finetune_epochs, batch_size=16, verbose=2,
    )
    finetuned_pred = model.predict([held_images[eval_idx], held_geos[eval_idx]], verbose=0)
    finetuned_metrics = evaluate_binary(held_labels[eval_idx], finetuned_pred)
    print(f"fine-tuned on {held_out}: {finetuned_metrics}")

    result = {
        "held_out_hurricane": held_out,
        "train_hurricanes": train_hurricanes,
        "n_train": len(train_samples),
        "n_held_out": len(held_out_samples),
        "n_finetune": len(ft_idx),
        "n_finetune_eval": len(eval_idx),
        "zero_shot_metrics": zero_shot_metrics,
        "finetuned_metrics": finetuned_metrics,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"results_{held_out}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--held-out", required=True, choices=HURRICANES)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--finetune-fraction", type=float, default=0.2)
    parser.add_argument("--max-samples-per-hurricane", type=int, default=None)
    parser.add_argument("--out-dir", default="/kaggle/working/exp_a_cross_hurricane")
    args = parser.parse_args()

    run(args.root, args.held_out, args.epochs, args.finetune_epochs,
        args.finetune_fraction, args.out_dir, args.max_samples_per_hurricane)
