"""
Reconstruction of the paper's model architecture (no source code was released
with the paper -- this is derived from the description and Fig. 3 layout in
PAPER.md's "Proposed Methodology" section).

v2 revision note: an earlier version collapsed the image to a 4-D vector
*before* the MobileNet-inspired block, leaving DepthwiseConv2D with no
spatial extent to operate on (a near-no-op on a 1x1 map). Empirically this
trained to a stuck ~55-58% accuracy (barely above chance) regardless of
learning-rate schedule, L2 regularization, or early stopping -- a real
architectural degeneracy, not a missing-hyperparameter issue. Fig. 3's
layout (Concatenate happens after the conv stack, not before it) supports a
different reading: the encoder's latent stays a genuine spatial feature map,
flows through real spatial Conv2D/DepthwiseConv2D/pooling layers (this is
what "MobileNetv2-inspired" should mean), and is only summarized to 4-D at
the very end, right before concatenating with the 4-D geo embedding.

  Image branch:  autoencoder encoder (Conv2D+ReLU+MaxPool x3) -> spatial
                  latent feature map -> UpSampling decoder (reconstruction
                  loss only, discarded at inference).
  Geo branch:    FC layers on (lat, lon) -> 4-D embedding.
  Classifier:    the spatial latent map flows through a MobileNetv2-inspired
                  stack (Conv2D, DepthwiseConv2D, BatchNorm, MaxPooling) with
                  a residual connection around the depthwise block, then
                  Flatten -> Dense(4) to summarize to a 4-D image embedding.
  Fusion:        concat(4-D image summary, 4-D geo embedding) -> 8-D ->
                  Dense -> Dropout -> single-node sigmoid (binary) or
                  4-node softmax (severity, Exp B).

Ambiguities the paper doesn't specify (input resolution, channel counts,
number of residual blocks) are resolved with documented, reasonable
defaults below -- flagged inline. This is a reconstruction to enable new
experiments, not a byte-exact reproduction of unreleased code.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

IMAGE_SIZE = (128, 128, 3)   # not stated in the paper; matches xbd_preprocess.py TARGET_SIZE
GEO_EMBED_DIM = 4            # stated: "FC layers ... yielding a 4-dimensional embedding"
IMAGE_SUMMARY_DIM = 4        # stated: concatenation of the two embeddings gives dim 8 total
FUSION_EMBED_DIM = IMAGE_SUMMARY_DIM + GEO_EMBED_DIM  # = 8, per the paper


def build_autoencoder_encoder(image_size=IMAGE_SIZE):
    """Conv2D+ReLU encoder -> spatial latent feature map (NOT flattened).
    Returns (input, latent_feature_map)."""
    inputs = layers.Input(shape=image_size, name="image_input")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    latent = layers.Conv2D(128, 3, padding="same", activation="relu", name="image_latent_map")(x)
    return inputs, latent


def build_autoencoder_decoder(latent_shape, image_size=IMAGE_SIZE):
    """UpSampling decoder, mirrors the encoder. Used only for the pretraining
    reconstruction loss -- discarded before the classifier is attached."""
    latent_input = layers.Input(shape=latent_shape, name="latent_input")
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(latent_input)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D()(x)
    outputs = layers.Conv2D(image_size[-1], 3, padding="same", activation="sigmoid")(x)
    return models.Model(latent_input, outputs, name="autoencoder_decoder")


def build_autoencoder(image_size=IMAGE_SIZE):
    """Full autoencoder (encoder+decoder) for the reconstruction-loss pretraining stage."""
    inputs, latent = build_autoencoder_encoder(image_size)
    latent_shape = latent.shape[1:]
    decoder = build_autoencoder_decoder(latent_shape, image_size)
    outputs = decoder(latent)
    autoencoder = models.Model(inputs, outputs, name="autoencoder")
    encoder = models.Model(inputs, latent, name="encoder")
    return autoencoder, encoder


def build_geo_branch(geo_dim=2, embed_dim=GEO_EMBED_DIM):
    """FC layers on (lat, lon) -> embed_dim embedding."""
    inputs = layers.Input(shape=(geo_dim,), name="geo_input")
    x = layers.Dense(16, activation="relu")(inputs)
    x = layers.Dense(embed_dim, activation="relu", name="geo_embedding")(x)
    return inputs, x


def build_image_classifier_path(latent_shape, use_residual=True, l2_reg=1e-4):
    """MobileNetv2-inspired stack operating on the encoder's spatial latent
    map -- real Conv2D/DepthwiseConv2D/pooling with spatial extent, unlike
    the degenerate 1x1-map version this replaces. Summarizes to a 4-D
    embedding matching IMAGE_SUMMARY_DIM."""
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    latent_input = layers.Input(shape=latent_shape, name="latent_map_input")

    x = layers.Conv2D(128, 3, padding="same")(latent_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    shortcut = x

    y = layers.DepthwiseConv2D(3, padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    if use_residual:
        y = layers.Add()([shortcut, y])

    x = layers.MaxPooling2D()(y)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    outputs = layers.Dense(IMAGE_SUMMARY_DIM, activation="relu", name="image_summary", kernel_regularizer=reg)(x)
    return latent_input, outputs


def build_classifier_head(fusion_dim=FUSION_EMBED_DIM, num_classes=1, dropout_rate=0.3, l2_reg=1e-4):
    """Final Dense -> Dropout -> output on the fused image+geo embedding.

    num_classes=1 -> binary sigmoid output (matches the original paper).
    num_classes=4 -> severity softmax output (Exp B, EXPERIMENT_DESIGN.md).
    """
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    fused_input = layers.Input(shape=(fusion_dim,), name="fused_embedding")
    x = layers.Dense(16, activation="relu", kernel_regularizer=reg)(fused_input)
    x = layers.Dropout(dropout_rate)(x)
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="binary_output", kernel_regularizer=reg)(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="severity_output", kernel_regularizer=reg)(x)
    return fused_input, outputs


def build_full_model(image_size=IMAGE_SIZE, num_classes=1, use_residual=True,
                      dropout_rate=0.3, l2_reg=1e-4):
    """End-to-end model: image encoder -> MobileNet-inspired spatial stack ->
    4-D image summary; geo branch -> 4-D geo embedding; concat -> 8-D fused
    representation -> classifier head.

    Set num_classes=1 to reproduce the original paper's binary task.
    Set num_classes=4 to run Exp B's severity extension.
    """
    image_input, image_latent_map = build_autoencoder_encoder(image_size)
    latent_shape = image_latent_map.shape[1:]

    classifier_path_input, image_summary = build_image_classifier_path(
        latent_shape, use_residual=use_residual, l2_reg=l2_reg
    )
    classifier_path_model = models.Model(classifier_path_input, image_summary, name="mobilenet_inspired_path")
    image_summary_out = classifier_path_model(image_latent_map)

    geo_input, geo_embedding = build_geo_branch(geo_dim=2, embed_dim=GEO_EMBED_DIM)
    fused = layers.Concatenate(name="fusion")([image_summary_out, geo_embedding])

    head_input, head_output = build_classifier_head(
        fusion_dim=FUSION_EMBED_DIM, num_classes=num_classes, dropout_rate=dropout_rate, l2_reg=l2_reg,
    )
    head_model = models.Model(head_input, head_output, name="classifier_head")
    outputs = head_model(fused)

    return models.Model([image_input, geo_input], outputs, name="autoencoder_c_mobilenetv2")


def _self_test():
    import numpy as np

    autoencoder, encoder = build_autoencoder()
    dummy_images = np.random.rand(2, *IMAGE_SIZE).astype("float32")
    recon = autoencoder.predict(dummy_images, verbose=0)
    assert recon.shape == dummy_images.shape, f"autoencoder shape mismatch: {recon.shape}"
    latent = encoder.predict(dummy_images, verbose=0)
    assert len(latent.shape) == 4 and latent.shape[-1] == 128, f"encoder latent should be a spatial map, got {latent.shape}"

    binary_model = build_full_model(num_classes=1, use_residual=True)
    dummy_geo = np.random.rand(2, 2).astype("float32")
    out = binary_model.predict([dummy_images, dummy_geo], verbose=0)
    assert out.shape == (2, 1), f"binary output shape wrong: {out.shape}"
    assert ((out >= 0) & (out <= 1)).all(), "sigmoid output out of [0,1] range"

    severity_model = build_full_model(num_classes=4, use_residual=True)
    out4 = severity_model.predict([dummy_images, dummy_geo], verbose=0)
    assert out4.shape == (2, 4), f"severity output shape wrong: {out4.shape}"
    row_sums = out4.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), f"softmax rows don't sum to 1: {row_sums}"

    no_res_model = build_full_model(num_classes=1, use_residual=False)
    out_nr = no_res_model.predict([dummy_images, dummy_geo], verbose=0)
    assert out_nr.shape == (2, 1), "non-residual variant shape wrong"

    n_params_binary = binary_model.count_params()
    n_params_severity = severity_model.count_params()
    assert n_params_severity > n_params_binary, "severity head should have more params (4 outputs vs 1)"

    # sanity: the classifier path must actually see spatial structure (H*W > 1)
    h, w = latent.shape[1], latent.shape[2]
    assert h > 1 and w > 1, f"encoder latent map collapsed to non-spatial shape: {latent.shape}"

    print(f"self-test OK -- latent map shape: {latent.shape}, binary params: {n_params_binary:,}, severity params: {n_params_severity:,}")


if __name__ == "__main__":
    _self_test()
