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
    return result


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
    for arch_name in ARCHITECTURES:
        print(f"=== training baseline: {arch_name} ===")
        all_results[arch_name] = run_one(arch_name, images, geos, labels, train_idx, test_idx, epochs, out_dir)

    with open(os.path.join(out_dir, "all_baselines.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", default="/kaggle/working/exp_c_baselines")
    args = parser.parse_args()

    run(args.root, args.epochs, args.out_dir, args.max_samples)
