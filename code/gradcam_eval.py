"""
Exp E: Grad-CAM attention maps for the trained binary model, plus a
quantitative localization metric (pointing-game accuracy).

Grad-CAM target layer: the autoencoder encoder's spatial latent map
("image_latent_map", 16x16x128) -- the last point in the network with
meaningful spatial resolution, since every layer after it in the
MobileNet-inspired path pools the map down further.

Pointing-game accuracy: for each test image, check whether the Grad-CAM
heatmap's peak-activation pixel falls within the center 50% region of the
crop. Every crop in this paper's datasets is built centered on one
building (xbd_preprocess.py's bounding-box crop, and the Harvey dataset's
existing per-building crops), so the center region is a reasonable
location proxy in the absence of a finer per-building ground-truth mask.
This is the standard pointing-game formulation (Zhang et al., 2018),
adapted here to a coarse center-region target rather than a segmentation
mask.

Run inside a Kaggle kernel with GPU enabled, dataset_sources including the
Harvey dataset, and kernel_sources pointing at the Exp C training run
(amanjolly1994/exp-c-harvey-train) for its saved model.keras.
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_harvey_samples, samples_to_arrays, normalize_geo


def build_gradcam_model(model, target_layer_name="image_latent_map"):
    """A model with the same inputs as `model`, outputting both the target
    layer's activation and the final prediction, so both come from one
    forward pass sharing the same computation graph."""
    target_output = model.get_layer(target_layer_name).output
    return tf.keras.Model(inputs=model.inputs, outputs=[target_output, model.output])


def compute_gradcam_batch(grad_model, images, geos):
    """Returns an (N, H, W) array of heatmaps in [0, 1], one per input."""
    images = tf.constant(images, dtype=tf.float32)
    geos = tf.constant(geos, dtype=tf.float32)

    with tf.GradientTape() as tape:
        latent_map, preds = grad_model([images, geos])
        tape.watch(latent_map)
        loss = preds[:, 0]

    grads = tape.gradient(loss, latent_map)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    weighted = latent_map * pooled_grads
    heatmaps = tf.reduce_sum(weighted, axis=-1)
    heatmaps = tf.nn.relu(heatmaps)
    max_per_image = tf.reduce_max(heatmaps, axis=(1, 2), keepdims=True)
    heatmaps = heatmaps / (max_per_image + 1e-8)
    return heatmaps.numpy()


def pointing_game_hit(heatmap, center_fraction=0.5):
    """True if the heatmap's argmax pixel falls within the center
    center_fraction x center_fraction region of the map."""
    h, w = heatmap.shape
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    y0, y1 = h * (1 - center_fraction) / 2, h * (1 + center_fraction) / 2
    x0, x1 = w * (1 - center_fraction) / 2, w * (1 + center_fraction) / 2
    return y0 <= peak_y <= y1 and x0 <= peak_x <= x1


def run(model_path, data_root, out_dir, n_samples=200, batch_size=32, seed=42):
    np.random.seed(seed)
    model = tf.keras.models.load_model(model_path)
    grad_model = build_gradcam_model(model)

    samples = load_harvey_samples(data_root)
    idx = np.random.choice(len(samples), size=min(n_samples, len(samples)), replace=False)
    samples = [samples[i] for i in idx]

    images, geos, labels = samples_to_arrays(samples, num_classes=1)
    geos = normalize_geo(geos)

    hits = []
    for start in range(0, len(images), batch_size):
        end = start + batch_size
        heatmaps = compute_gradcam_batch(grad_model, images[start:end], geos[start:end])
        hits.extend(pointing_game_hit(h) for h in heatmaps)

    pointing_game_accuracy = float(np.mean(hits))
    result = {
        "n_samples": len(images),
        "pointing_game_accuracy": pointing_game_accuracy,
        "target_layer": "image_latent_map",
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gradcam_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="/kaggle/working/exp_e_gradcam")
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()

    run(args.model_path, args.data_root, args.out_dir, args.n_samples)
