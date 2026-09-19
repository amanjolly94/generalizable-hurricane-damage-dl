"""
Train the 2022-era transfer-learning baselines (VGG16, MobileNetV2,
DenseNet121 -- the same three Kaur et al. [12] compare) on the Harvey
dataset, image-only (no geo branch, matching how those baselines are
defined in the cited literature), for Exp C's efficiency table.

A 2023+ transformer baseline (Lu et al. 2024 Bitemporal Attention
Transformer) is deferred to a follow-up run -- it needs a from-scratch
implementation, not a tf.keras.applications drop-in, and shouldn't be
rushed to hit a parallelism window.

Run inside a Kaggle kernel with GPU enabled and
dataset_sources: ["kmader/satellite-images-of-hurricane-damage"].
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import IMAGE_SIZE
import harvey_preprocess
from train import load_harvey_samples, samples_to_arrays, evaluate_binary, compute_flops, measure_latency

ARCHITECTURES = {
    "VGG16": tf.keras.applications.VGG16,
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "DenseNet121": tf.keras.applications.DenseNet121,
}


def build_baseline(arch_name, image_size=IMAGE_SIZE, freeze_base=True):
    base_cls = ARCHITECTURES[arch_name]
    base = base_cls(include_top=False, weights="imagenet", input_shape=image_size, pooling="avg")
    base.trainable = not freeze_base
    inputs = layers.Input(shape=image_size)
    x = base(inputs, training=not freeze_base)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name=f"{arch_name}_baseline")


def run_one(arch_name, images, geos, labels, train_idx, test_idx, epochs, out_dir):
    model = build_baseline(arch_name)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        images[train_idx], labels[train_idx].astype("float32"),
        validation_split=0.1, epochs=epochs, batch_size=32, verbose=2,
    )
    y_pred_prob = model.predict(images[test_idx], verbose=0)
    metrics = evaluate_binary(labels[test_idx], y_pred_prob)

    n_params = model.count_params()
    flops = compute_flops(model)
    latency_s = measure_latency(model, images[test_idx][:1])

    result = {
        "architecture": arch_name,
        "metrics": metrics,
        "n_params": int(n_params),
        "flops": flops,
        "latency_ms_per_image": latency_s * 1000,
    }
    arch_dir = os.path.join(out_dir, arch_name)
    os.makedirs(arch_dir, exist_ok=True)
    with open(os.path.join(arch_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result, y_pred_prob.ravel()


def save_combined_predictions(path, samples, test_idx, labels, all_probs, arch_names):
    """One predictions file across all baselines, keyed by source image path
    so it can be joined with the proposed model's own predictions.csv
    (train.py's save_predictions) for a paired test (McNemar's) or a
    threshold sweep -- both need per-sample predictions from every model on
    the identical test items, which aggregate accuracy alone can't provide."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label"] + [f"{a}_prob" for a in arch_names])
        for i, idx in enumerate(test_idx):
            row = [samples[idx][0], int(labels[test_idx][i])]
            row += [float(all_probs[a][i]) for a in arch_names]
            writer.writerow(row)


def run(root, epochs, out_dir, max_samples=None, seed=42):
    np.random.seed(seed)
    samples = load_harvey_samples(root)
    if max_samples:
        idx = np.random.choice(len(samples), size=min(max_samples, len(samples)), replace=False)
        samples = [samples[i] for i in idx]
    print(f"loaded {len(samples)} Harvey samples for baseline comparison")

    images, geos, labels = samples_to_arrays(samples, num_classes=1)
    n = len(images)
    perm = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = perm[:split], perm[split:]

    os.makedirs(out_dir, exist_ok=True)
    all_results = {}
    all_probs = {}
    for arch_name in ARCHITECTURES:
        print(f"=== training baseline: {arch_name} ===")
        all_results[arch_name], all_probs[arch_name] = run_one(
            arch_name, images, geos, labels, train_idx, test_idx, epochs, out_dir
        )

    with open(os.path.join(out_dir, "all_baselines.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    save_combined_predictions(
        os.path.join(out_dir, "predictions.csv"), samples, test_idx, labels, all_probs, list(ARCHITECTURES)
    )

    return all_results


def _self_test():
    import tempfile

    samples = [(f"/fake/img{i}.png", 0.0, 0.0, i % 2, i % 4) for i in range(6)]
    test_idx = [1, 3, 5]
    labels = np.array([0, 1, 0, 1, 0, 1])
    all_probs = {
        "VGG16": np.array([0.9, 0.2, 0.6]),
        "MobileNetV2": np.array([0.8, 0.3, 0.5]),
        "DenseNet121": np.array([0.7, 0.4, 0.4]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "predictions.csv")
        save_combined_predictions(path, samples, test_idx, labels, all_probs, list(ARCHITECTURES))
        with open(path) as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["path", "true_label", "VGG16_prob", "MobileNetV2_prob", "DenseNet121_prob"]
        assert rows[1] == ["/fake/img1.png", "1", "0.9", "0.8", "0.7"]
        assert len(rows) == 4

    print("self-test OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", default="/kaggle/working/exp_c_baselines")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
    else:
        if not args.root:
            parser.error("--root is required unless --self-test is passed")
        run(args.root, args.epochs, args.out_dir, args.max_samples)
