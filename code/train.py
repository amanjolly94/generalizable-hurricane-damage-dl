"""
Train code/model.py's reconstructed architecture on either the Harvey
dataset (binary, Exp C groundwork) or the pooled xBD hurricane patches
(binary or severity, Exp A/B groundwork).

Reports accuracy/precision/recall/F1 (or per-class for severity), parameter
count, best-effort FLOPs, and mean single-image inference latency -- the
Exp C efficiency-table columns -- alongside the trained model.

Run inside a Kaggle kernel with GPU enabled. For --dataset harvey:
  dataset_sources: ["kmader/satellite-images-of-hurricane-damage"]
For --dataset xbd:
  kernel_sources: ["amanjolly1994/xbd-hurricane-preprocess"]
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_full_model, IMAGE_SIZE

import harvey_preprocess


def load_harvey_samples(root, splits=("train_another", "validation_another")):
    samples = []
    for split, label_name, fname, full_path in harvey_preprocess.iter_images(root):
        if split not in splits:
            continue
        parsed = harvey_preprocess.parse_lon_lat_from_filename(fname)
        if parsed is None:
            continue
        lat, lon = parsed
        label = harvey_preprocess.LABEL_MAP[label_name]
        samples.append((full_path, lat, lon, label, label))  # (path, lat, lon, binary, severity=binary placeholder)
    return samples


def load_xbd_samples(root):
    """root = the mounted preprocessing-kernel output dir containing
    metadata.csv and patches/."""
    metadata_path = os.path.join(root, "metadata.csv")
    patches_dir = os.path.join(root, "patches")
    samples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            full_path = os.path.join(patches_dir, row["patch_filename"])
            if not os.path.exists(full_path):
                continue
            samples.append((
                full_path,
                float(row["lat"]),
                float(row["lon"]),
                int(row["binary_label"]),
                int(row["severity_label"]),
            ))
    return samples


def samples_to_arrays(samples, num_classes, image_size=IMAGE_SIZE):
    images = np.zeros((len(samples), *image_size), dtype="float32")
    geos = np.zeros((len(samples), 2), dtype="float32")
    labels = np.zeros((len(samples),), dtype="int64")
    for i, (path, lat, lon, binary_label, severity_label) in enumerate(samples):
        img = Image.open(path).convert("RGB").resize(image_size[:2], Image.BILINEAR)
        images[i] = np.asarray(img, dtype="float32") / 255.0
        geos[i] = [lat, lon]
        labels[i] = severity_label if num_classes == 4 else binary_label
    return images, geos, labels


def sparse_categorical_focal_loss(gamma=2.0):
    """Focal loss (Lin et al. 2017) for sparse integer labels, manually
    implemented to avoid an extra pip dependency. Down-weights easy/majority
    examples so the loss doesn't stay dominated by the classes the model
    already gets right -- the standard fix for the exact collapse observed
    in Exp B's plain-cross-entropy baseline (model never predicts the two
    minority severity classes)."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        probs = tf.gather(y_pred, y_true, batch_dims=1)
        focal_weight = tf.pow(1.0 - probs, gamma)
        ce = -tf.math.log(probs)
        return focal_weight * ce
    return loss_fn


def normalize_geo(geos):
    mean = geos.mean(axis=0, keepdims=True)
    std = geos.std(axis=0, keepdims=True) + 1e-8
    return (geos - mean) / std


def compute_flops(model):
    """Best-effort FLOPs count via keras-flops; None if unavailable or if the
    package's TF op-statistics registration collides with this kernel's TF
    version (observed: KeyError on 'AddV2,flops' double-registration) --
    never let a FLOPs failure lose an already-trained model's real results."""
    try:
        try:
            from keras_flops import get_flops
        except ImportError:
            import subprocess
            subprocess.run(["pip", "install", "-q", "keras-flops"], check=False)
            from keras_flops import get_flops
        return get_flops(model, batch_size=1)
    except Exception as e:
        print(f"FLOPs computation unavailable ({e}); skipping.")
        return None


def measure_latency(model, sample_inputs, n_runs=50):
    # warmup
    for _ in range(5):
        model.predict(sample_inputs, verbose=0)
    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(sample_inputs, verbose=0)
    elapsed = time.perf_counter() - start
    return elapsed / n_runs


def evaluate_binary(y_true, y_pred_prob):
    y_pred = (y_pred_prob.ravel() >= 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def evaluate_severity(y_true, y_pred_prob, num_classes=4):
    y_pred = y_pred_prob.argmax(axis=1)
    acc = float((y_pred == y_true).mean())
    per_class = {}
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": int((y_true == c).sum())}
    confusion = [[int(((y_true == i) & (y_pred == j)).sum()) for j in range(num_classes)] for i in range(num_classes)]
    return {"accuracy": acc, "per_class": per_class, "confusion_matrix": confusion}


def run(dataset, root, num_classes, epochs, out_dir, max_samples=None, seed=42, use_focal_loss=False):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)

    if dataset == "harvey":
        samples = load_harvey_samples(root)
    elif dataset == "xbd":
        samples = load_xbd_samples(root)
    else:
        raise ValueError(dataset)

    if max_samples:
        idx = np.random.choice(len(samples), size=min(max_samples, len(samples)), replace=False)
        samples = [samples[i] for i in idx]

    print(f"loaded {len(samples)} samples for dataset={dataset}, num_classes={num_classes}")

    images, geos, labels = samples_to_arrays(samples, num_classes)
    geos = normalize_geo(geos)

    n = len(images)
    perm = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = perm[:split], perm[split:]

    model = build_full_model(num_classes=num_classes, use_residual=True)
    if num_classes == 1:
        loss = "binary_crossentropy"
    elif use_focal_loss:
        loss = sparse_categorical_focal_loss(gamma=2.0)
    else:
        loss = "sparse_categorical_crossentropy"
    # Paper states Adam lr=1e-2. Empirically, on two independent from-scratch
    # architecture reconstructions, lr=1e-2 produced a flat, non-learning loss
    # curve from epoch 1 (~0.65, never improving) -- 1e-2 is unusually high
    # for Adam on a from-scratch small CNN (SGD-range, not typical Adam
    # range); the paper almost certainly relied on an unstated detail
    # (warmup, gradient clipping) that doesn't transfer to this
    # reconstruction. Using the standard Adam default (1e-3) instead, with
    # the paper's decay-to-80%-every-3-epochs schedule and early stopping
    # kept. Documented here as a deliberate, evidence-based deviation from
    # the paper's stated value, not an oversight.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, lr: lr * 0.8 if epoch > 0 and epoch % 3 == 0 else lr
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    y_train = labels[train_idx].astype("float32") if num_classes == 1 else labels[train_idx]
    model.fit(
        [images[train_idx], geos[train_idx]], y_train,
        validation_split=0.1, epochs=epochs, batch_size=32, verbose=2,
        callbacks=[lr_schedule, early_stopping],
    )

    y_pred_prob = model.predict([images[test_idx], geos[test_idx]], verbose=0)
    y_test = labels[test_idx]

    if num_classes == 1:
        metrics = evaluate_binary(y_test, y_pred_prob)
    else:
        metrics = evaluate_severity(y_test, y_pred_prob, num_classes=num_classes)

    n_params = model.count_params()
    flops = compute_flops(model)
    sample_inputs = [images[test_idx][:1], geos[test_idx][:1]]
    latency_s = measure_latency(model, sample_inputs)

    result = {
        "dataset": dataset,
        "num_classes": num_classes,
        "loss_type": "focal" if (num_classes > 1 and use_focal_loss) else ("binary_ce" if num_classes == 1 else "categorical_ce"),
        "n_samples": n,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "epochs": epochs,
        "metrics": metrics,
        "n_params": int(n_params),
        "flops": flops,
        "latency_ms_per_image": latency_s * 1000,
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    model.save(os.path.join(out_dir, "model.keras"))

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["harvey", "xbd"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--num-classes", type=int, choices=[1, 4], default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", default="/kaggle/working/train_output")
    parser.add_argument("--use-focal-loss", action="store_true")
    args = parser.parse_args()

    run(args.dataset, args.root, args.num_classes, args.epochs, args.out_dir, args.max_samples,
        use_focal_loss=args.use_focal_loss)
