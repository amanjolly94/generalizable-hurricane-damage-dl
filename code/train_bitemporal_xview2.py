"""
Experiment F: train the bitemporal variant of the architecture
(model.build_full_model_bitemporal) on the full xView2/xBD dataset (all 17
disaster types, not just the 4 hurricanes) and evaluate four-class severity
accuracy and macro F1, for a head-to-head comparison against Zarski &
Miszczak (2024, IEEE Access, DOI 10.1109/ACCESS.2024.3459424), who report
89.8% accuracy / 72.0% macro F1 on the same underlying dataset using a
bitemporal (pre+post disaster image) fusion network.

Their exact train/test split is not published, so this experiment uses its
own 80/20 split -- the comparison below is against their reported numbers,
not a split-identical reproduction, and that limitation is reported alongside
whatever result this produces.

At full-xView2 scale (~107,000 building pairs), loading every image into a
single float32 numpy array up front needs roughly 42 GB of RAM (two full
128x128x3 image arrays) and gets the process killed by the OOM killer before
training even starts -- confirmed empirically on a first run of this script.
Images are streamed through a tf.data pipeline instead (decode+resize
per-batch, on demand), so at most a few batches of images are ever resident
in memory at once.

Reuses sparse_categorical_focal_loss, evaluate_severity, compute_flops, and
measure_latency from train.py rather than duplicating them.

Run inside a Kaggle kernel with GPU enabled, after preprocessing xBD with:
  python xbd_preprocess.py --all-disasters --include-pre-disaster ...

  kernel_sources: ["<owner>/<xbd-all-disasters-bitemporal-preprocess-slug>"]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_full_model_bitemporal, build_full_model_bitemporal_pretrained, IMAGE_SIZE
from train import (
    sparse_categorical_focal_loss,
    evaluate_severity,
    compute_flops,
    measure_latency,
)


def load_bitemporal_xbd_samples(roots):
    """roots = one or more mounted preprocessing-kernel output dirs, each
    containing metadata.csv (with a pre_patch_filename column) and patches/.
    Experiment F's preprocessing runs as several parallel shards (one per
    Kaggle account) that each cover a disjoint slice of xView2, so training
    reads all of them and pools the samples -- a single root still works,
    passed as a one-element list."""
    if isinstance(roots, str):
        roots = [roots]
    samples = []
    for root in roots:
        metadata_path = os.path.join(root, "metadata.csv")
        patches_dir = os.path.join(root, "patches")
        n_root = 0
        with open(metadata_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pre_filename = row.get("pre_patch_filename")
                if not pre_filename:
                    continue
                post_path = os.path.join(patches_dir, row["patch_filename"])
                pre_path = os.path.join(patches_dir, pre_filename)
                if not (os.path.exists(post_path) and os.path.exists(pre_path)):
                    continue
                samples.append((
                    pre_path, post_path,
                    float(row["lat"]), float(row["lon"]),
                    int(row["severity_label"]),
                ))
                n_root += 1
        print(f"loaded {n_root} samples from {root}")
    return samples


def _augment_pair(pre, post):
    """Applies the same random flip/rotation to both images in a pair, so the
    bitemporal correspondence between them survives the augmentation (a flip
    applied to only one of the two would misalign pre- and post-disaster
    building geometry). Matches the flip/rotation part of Zarski &
    Miszczak's (2024) own augmentation set (random flips, rotations, affine,
    Gaussian blur, Gaussian noise); this only reproduces the flip/rotation
    piece, not the full set."""
    if tf.random.uniform([]) < 0.5:
        pre = tf.image.flip_left_right(pre)
        post = tf.image.flip_left_right(post)
    if tf.random.uniform([]) < 0.5:
        pre = tf.image.flip_up_down(pre)
        post = tf.image.flip_up_down(post)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    pre = tf.image.rot90(pre, k)
    post = tf.image.rot90(post, k)
    return pre, post


def _make_dataset(samples, indices, geo_mean, geo_std, batch_size,
                   shuffle, image_size=IMAGE_SIZE, seed=42, augment=False,
                   mobilenet_preprocessing=False):
    """Builds a tf.data pipeline that decodes and resizes each pre/post image
    pair lazily, per batch, instead of materializing the whole split as a
    numpy array. Yields ((pre_image, post_image, geo), label) batches,
    matching build_full_model_bitemporal's [pre_input, post_input, geo_input]
    input order. augment=True applies random flips/rotations (training split
    only -- never on validation/test, where evaluation must stay on the
    real, unmodified images).

    mobilenet_preprocessing=True scales pixels to [-1, 1] via
    tf.keras.applications.mobilenet_v2.preprocess_input instead of the
    default [0, 1] scaling -- required when feeding
    build_full_model_bitemporal_pretrained's frozen ImageNet-pretrained
    MobileNetV2 backbones, which were trained on [-1, 1]-scaled inputs.
    Feeding them [0, 1]-scaled images instead would silently look like a
    working pipeline while actually breaking the pretrained features."""
    pre_paths = np.array([samples[i][0] for i in indices])
    post_paths = np.array([samples[i][1] for i in indices])
    geos = np.array([[samples[i][2], samples[i][3]] for i in indices], dtype="float32")
    geos = (geos - geo_mean) / geo_std
    labels = np.array([samples[i][4] for i in indices], dtype="int64")

    def _load_pair(pre_path, post_path, geo, label):
        def _load_one(path):
            raw = tf.io.read_file(path)
            img = tf.image.decode_png(raw, channels=3)
            img = tf.image.resize(img, image_size[:2])
            img = tf.cast(img, tf.float32)
            if mobilenet_preprocessing:
                return tf.keras.applications.mobilenet_v2.preprocess_input(img)
            return img / 255.0
        pre, post = _load_one(pre_path), _load_one(post_path)
        if augment:
            pre, post = _augment_pair(pre, post)
        return (pre, post, geo), label

    ds = tf.data.Dataset.from_tensor_slices((pre_paths, post_paths, geos, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(indices), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# Published comparator this experiment is measured against (Zarski &
# Miszczak 2024, Table 7, ResNet50 dual-branch, xView2 column).
COMPARATOR = {
    "name": "Zarski & Miszczak (2024, IEEE Access)",
    "doi": "10.1109/ACCESS.2024.3459424",
    "accuracy": 0.898,
    "macro_f1": 0.720,
    "dataset_note": "all 17 xView2 disaster types, 129,980 buildings, 74.9% majority class",
}


def run(root, epochs, out_dir, max_samples=None, seed=42, use_focal_loss=True, batch_size=32,
        share_encoder=True, l2_reg=1e-4, fusion_mode="concat", augment=False,
        model_variant="lightweight", share_backbone=False, fine_tune_backbone=False,
        learning_rate=1e-3, image_size=None):
    """root: a single shard directory, or a list of shard directories (all
    of Experiment F's parallel preprocessing shards pooled into one dataset).

    model_variant="lightweight" (default) trains build_full_model_bitemporal
    from scratch -- the paper's main, small (~1.5M param) architecture.
    model_variant="pretrained" trains build_full_model_bitemporal_pretrained
    instead: two ImageNet-pretrained MobileNetV2 backbones fused at multiple
    depths. share_encoder/l2_reg/fusion_mode apply only to "lightweight";
    share_backbone/fine_tune_backbone apply only to "pretrained".

    image_size defaults to model.IMAGE_SIZE (128x128x3), the resolution the
    source patches are already saved at. Passing a larger size (e.g.
    (224, 224, 3), MobileNetV2's native ImageNet training resolution)
    upscales the same 128x128 patches via bilinear resize -- this adds no
    new pixel detail, but lets the pretrained variant use MobileNetV2's
    native-resolution checkpoint and receptive field instead of the
    resolution-rescaled 128px one."""
    if image_size is None:
        image_size = IMAGE_SIZE
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)

    samples = load_bitemporal_xbd_samples(root)
    if max_samples:
        idx = np.random.choice(len(samples), size=min(max_samples, len(samples)), replace=False)
        samples = [samples[i] for i in idx]
    n = len(samples)
    print(f"loaded {n} bitemporal samples total")

    perm = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx_full, test_idx = perm[:split], perm[split:]
    val_split = int(len(train_idx_full) * 0.9)
    train_idx, val_idx = train_idx_full[:val_split], train_idx_full[val_split:]

    # Geo normalization stats from the training split only, to avoid leaking
    # test-set distribution into the model's input scaling.
    train_geos_raw = np.array([[samples[i][2], samples[i][3]] for i in train_idx], dtype="float32")
    geo_mean = train_geos_raw.mean(axis=0, keepdims=True)
    geo_std = train_geos_raw.std(axis=0, keepdims=True) + 1e-8

    mobilenet_preprocessing = (model_variant == "pretrained")
    train_ds = _make_dataset(samples, train_idx, geo_mean, geo_std, batch_size,
                              shuffle=True, seed=seed, augment=augment, image_size=image_size,
                              mobilenet_preprocessing=mobilenet_preprocessing)
    val_ds = _make_dataset(samples, val_idx, geo_mean, geo_std, batch_size, shuffle=False,
                            image_size=image_size, mobilenet_preprocessing=mobilenet_preprocessing)
    test_ds = _make_dataset(samples, test_idx, geo_mean, geo_std, batch_size, shuffle=False,
                             image_size=image_size, mobilenet_preprocessing=mobilenet_preprocessing)

    if model_variant == "lightweight":
        model = build_full_model_bitemporal(num_classes=4, use_residual=True,
                                             share_encoder=share_encoder, l2_reg=l2_reg,
                                             fusion_mode=fusion_mode)
    elif model_variant == "pretrained":
        model = build_full_model_bitemporal_pretrained(
            num_classes=4, l2_reg=l2_reg, image_size=image_size,
            share_backbone=share_backbone, fine_tune_backbone=fine_tune_backbone,
        )
    else:
        raise ValueError(f"unknown model_variant: {model_variant!r}")
    loss = sparse_categorical_focal_loss(gamma=2.0) if use_focal_loss else "sparse_categorical_crossentropy"

    # Same deliberate lr=1e-3 deviation documented in train.py -- lr=1e-2
    # produced a flat, non-learning loss on this architecture family. When
    # fine_tune_backbone=True, pass a much smaller learning_rate (e.g. 1e-5)
    # explicitly -- 1e-3 is standard for training a fresh head from scratch
    # but is large enough to destroy pretrained ImageNet weights during
    # fine-tuning, a well-known transfer-learning failure mode.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, lr: lr * 0.8 if epoch > 0 and epoch % 3 == 0 else lr
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, verbose=2,
        callbacks=[lr_schedule, early_stopping],
    )

    y_pred_prob = model.predict(test_ds, verbose=0)
    y_test = np.array([samples[i][4] for i in test_idx], dtype="int64")
    metrics = evaluate_severity(y_test, y_pred_prob, num_classes=4)
    macro_f1 = float(np.mean([metrics["per_class"][c]["f1"] for c in range(4)]))

    n_params = model.count_params()
    n_trainable = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    flops = compute_flops(model)

    # One real sample pair for latency timing, loaded directly (no need to
    # pull a whole batch through the tf.data pipeline just for this).
    pre_path0, post_path0, lat0, lon0, _ = samples[test_idx[0]]
    pre0_img = np.asarray(Image.open(pre_path0).convert("RGB").resize(image_size[:2], Image.BILINEAR), dtype="float32")
    post0_img = np.asarray(Image.open(post_path0).convert("RGB").resize(image_size[:2], Image.BILINEAR), dtype="float32")
    if mobilenet_preprocessing:
        pre0 = tf.keras.applications.mobilenet_v2.preprocess_input(pre0_img)[None]
        post0 = tf.keras.applications.mobilenet_v2.preprocess_input(post0_img)[None]
    else:
        pre0 = pre0_img[None] / 255.0
        post0 = post0_img[None] / 255.0
    geo0 = ((np.array([[lat0, lon0]], dtype="float32") - geo_mean) / geo_std)
    latency_s = measure_latency(model, [pre0, post0, geo0])

    result = {
        "experiment": "F_bitemporal_full_xview2",
        "model_variant": model_variant,
        "share_encoder": share_encoder,
        "l2_reg": l2_reg,
        "fusion_mode": fusion_mode,
        "augment": augment,
        "share_backbone": share_backbone,
        "fine_tune_backbone": fine_tune_backbone,
        "learning_rate": learning_rate,
        "image_size": list(image_size),
        "loss_type": "focal" if use_focal_loss else "categorical_ce",
        "n_samples": n,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "epochs": epochs,
        "metrics": metrics,
        "macro_f1": macro_f1,
        "n_params": int(n_params),
        "n_trainable_params": int(n_trainable),
        "flops": flops,
        "latency_ms_per_image": latency_s * 1000,
        "comparator": COMPARATOR,
        "beats_comparator_accuracy": metrics["accuracy"] > COMPARATOR["accuracy"],
        "beats_comparator_macro_f1": macro_f1 > COMPARATOR["macro_f1"],
        "split_note": "Own 80/20 split; Zarski & Miszczak's exact split is not published, so this is not a split-identical reproduction.",
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    model.save(os.path.join(out_dir, "model.keras"))

    print(json.dumps(result, indent=2))
    return result


def _self_test():
    """No real xBD data needed -- builds synthetic shards and verifies the
    streaming pipeline runs end to end without materializing full-dataset
    image arrays (the failure mode this rewrite fixes)."""
    import csv as _csv
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        roots = []
        for shard in range(2):
            root = os.path.join(tmp, f"shard{shard}")
            patches = os.path.join(root, "patches")
            os.makedirs(patches)
            rows = []
            for i in range(15):
                pre_name = f"s{shard}_{i}_pre.png"
                post_name = f"s{shard}_{i}_post.png"
                Image.new("RGB", (32, 32), color=(i * 10 % 255, 0, 0)).save(os.path.join(patches, pre_name))
                Image.new("RGB", (32, 32), color=(0, i * 10 % 255, 0)).save(os.path.join(patches, post_name))
                rows.append({
                    "patch_filename": post_name, "pre_patch_filename": pre_name,
                    "lat": 30.0 + i * 0.01, "lon": -90.0 - i * 0.01, "severity_label": i % 4,
                })
            with open(os.path.join(root, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=["patch_filename", "pre_patch_filename", "lat", "lon", "severity_label"])
                w.writeheader()
                w.writerows(rows)
            roots.append(root)

        result = run(roots, epochs=1, out_dir=os.path.join(tmp, "out"), seed=1, batch_size=4)
        assert result["n_samples"] == 30, f"expected 30 pooled samples across 2 shards, got {result['n_samples']}"
        assert result["n_train"] + result["n_val"] + result["n_test"] == 30, "train/val/test split doesn't cover all samples"
        assert 0.0 <= result["macro_f1"] <= 1.0, f"macro_f1 out of range: {result['macro_f1']}"
        assert os.path.exists(os.path.join(tmp, "out", "results.json"))

        # augment=True must not crash the pipeline, and must still cover every sample.
        result_aug = run(roots, epochs=1, out_dir=os.path.join(tmp, "out_aug"), seed=1, batch_size=4, augment=True)
        assert result_aug["n_samples"] == 30
        assert result_aug["augment"] is True

        # model_variant="pretrained" exercises the ImageNet-MobileNetV2 path,
        # including the [-1, 1] preprocessing switch -- must run end to end
        # without crashing on the mismatched preprocessing this rewrite guards against.
        result_pretrained = run(roots, epochs=1, out_dir=os.path.join(tmp, "out_pretrained"),
                                 seed=1, batch_size=4, model_variant="pretrained")
        assert result_pretrained["model_variant"] == "pretrained"
        assert result_pretrained["n_trainable_params"] < result_pretrained["n_params"], (
            "pretrained variant should have most of its parameters frozen"
        )

        # image_size=(224,224,3) upscales the same 128x128 source patches --
        # must run end to end without shape errors at MobileNetV2's native resolution.
        result_224 = run(roots, epochs=1, out_dir=os.path.join(tmp, "out_224"),
                          seed=1, batch_size=4, model_variant="pretrained", image_size=(224, 224, 3))
        assert result_224["image_size"] == [224, 224, 3]

    # _augment_pair must apply the SAME transform to both images in a pair --
    # a flip applied to only one would break the pre/post correspondence.
    # Use a non-symmetric pattern so a mismatched flip/rotation is detectable.
    pattern = np.zeros((4, 4, 1), dtype="float32")
    pattern[0, 0, 0] = 1.0
    pre = tf.constant(pattern)
    post = tf.constant(pattern.copy())
    for _ in range(20):
        aug_pre, aug_post = _augment_pair(pre, post)
        assert np.array_equal(aug_pre.numpy(), aug_post.numpy()), (
            "augmentation applied different transforms to the pre/post pair -- "
            "this would misalign bitemporal building geometry"
        )

    print("self-test OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append",
                         help="A shard's data directory; repeat --root for each shard to pool them.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", default="/kaggle/working/train_output_bitemporal")
    parser.add_argument("--no-focal-loss", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--share-encoder", dest="share_encoder", action="store_true", default=True,
                         help="One siamese encoder for both pre/post images (default).")
    parser.add_argument("--no-share-encoder", dest="share_encoder", action="store_false",
                         help="Two independently-weighted encoders, one per pre/post branch.")
    parser.add_argument("--l2-reg", type=float, default=1e-4)
    parser.add_argument("--fusion-mode", choices=["concat", "post_and_diff"], default="concat")
    parser.add_argument("--augment", action="store_true",
                         help="Random flips/rotations on the training split (not val/test).")
    parser.add_argument("--model-variant", choices=["lightweight", "pretrained"], default="lightweight",
                         help="'lightweight': build_full_model_bitemporal (from scratch). "
                              "'pretrained': build_full_model_bitemporal_pretrained (ImageNet MobileNetV2 backbones).")
    parser.add_argument("--share-backbone", action="store_true",
                         help="pretrained variant only: one shared MobileNetV2 backbone instead of two independent ones.")
    parser.add_argument("--fine-tune-backbone", action="store_true",
                         help="pretrained variant only: unfreeze the MobileNetV2 backbones instead of the default frozen-feature-extractor setting.")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                         help="Use a much smaller value (e.g. 1e-5) with --fine-tune-backbone to avoid destroying pretrained weights.")
    parser.add_argument("--image-size", type=int, default=None,
                         help="Square input resolution (e.g. 224 for MobileNetV2's native ImageNet resolution). Defaults to the source patch resolution (128).")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
    else:
        if not args.root:
            parser.error("--root is required unless --self-test is passed")
        image_size = (args.image_size, args.image_size, 3) if args.image_size else None
        run(args.root, args.epochs, args.out_dir, args.max_samples,
            use_focal_loss=not args.no_focal_loss, batch_size=args.batch_size,
            share_encoder=args.share_encoder, l2_reg=args.l2_reg, fusion_mode=args.fusion_mode,
            augment=args.augment, model_variant=args.model_variant,
            share_backbone=args.share_backbone, fine_tune_backbone=args.fine_tune_backbone,
            learning_rate=args.learning_rate, image_size=image_size)
