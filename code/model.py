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


def build_image_classifier_path(latent_shape, use_residual=True, l2_reg=1e-4, image_summary_dim=IMAGE_SUMMARY_DIM):
    """MobileNetv2-inspired stack operating on the encoder's spatial latent
    map -- real Conv2D/DepthwiseConv2D/pooling with spatial extent, unlike
    the degenerate 1x1-map version this replaces. Summarizes to a
    image_summary_dim embedding (4-D by default, matching the paper's
    stated 4+4=8 fusion width for the binary task; widened for the
    severity task -- see build_full_model's num_classes>1 branch)."""
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
    outputs = layers.Dense(image_summary_dim, activation="relu", name="image_summary", kernel_regularizer=reg)(x)
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
                      dropout_rate=0.3, l2_reg=1e-4, image_summary_dim=None):
    """End-to-end model: image encoder -> MobileNet-inspired spatial stack ->
    image summary; geo branch -> 4-D geo embedding; concat -> fused
    representation -> classifier head.

    Set num_classes=1 to reproduce the original paper's binary task (uses
    the paper's stated 4-D image summary -> 8-D fusion width).

    Set num_classes=4 for Exp B's severity extension. image_summary_dim
    defaults to 16 (not 4) for num_classes>1: an 8-D fused vector (4+4)
    was found to be an information bottleneck for 4-class severity --
    both cross-entropy and focal loss converged to byte-identical
    predictions that only ever separated 2 of the 4 classes, regardless of
    loss function, which is the signature of a capacity ceiling rather
    than an optimization problem. Pass image_summary_dim explicitly to
    override either default.
    """
    if image_summary_dim is None:
        image_summary_dim = IMAGE_SUMMARY_DIM if num_classes == 1 else 16

    image_input, image_latent_map = build_autoencoder_encoder(image_size)
    latent_shape = image_latent_map.shape[1:]

    classifier_path_input, image_summary = build_image_classifier_path(
        latent_shape, use_residual=use_residual, l2_reg=l2_reg, image_summary_dim=image_summary_dim
    )
    classifier_path_model = models.Model(classifier_path_input, image_summary, name="mobilenet_inspired_path")
    image_summary_out = classifier_path_model(image_latent_map)

    geo_input, geo_embedding = build_geo_branch(geo_dim=2, embed_dim=GEO_EMBED_DIM)
    fused = layers.Concatenate(name="fusion")([image_summary_out, geo_embedding])

    fusion_dim = image_summary_dim + GEO_EMBED_DIM
    head_input, head_output = build_classifier_head(
        fusion_dim=fusion_dim, num_classes=num_classes, dropout_rate=dropout_rate, l2_reg=l2_reg,
    )
    head_model = models.Model(head_input, head_output, name="classifier_head")
    outputs = head_model(fused)

    return models.Model([image_input, geo_input], outputs, name="autoencoder_c_mobilenetv2")


def build_full_model_bitemporal(image_size=IMAGE_SIZE, num_classes=1, use_residual=True,
                                 dropout_rate=0.3, l2_reg=1e-4, image_summary_dim=None,
                                 share_encoder=True, fusion_mode="concat"):
    """Bitemporal variant for Experiment F (EXPERIMENT_DESIGN.md): takes both
    a pre-disaster and a post-disaster image instead of a single post-disaster
    image, so the comparison against Zarski & Miszczak (2024)'s bitemporal
    fusion baseline on full xView2 is a fair test of the same input modality,
    not just a class-imbalance difference.

    share_encoder=True (the first version tested) runs one autoencoder
    encoder (siamese) on both the pre- and post-disaster images.
    share_encoder=False builds two independently-weighted encoders instead --
    Zarski & Miszczak's own ablation (their ResNet50D vs. ResNet50S, Table 7)
    found two separate feature-extraction paths outperform one shared path
    on this exact task, since pre- and post-disaster imagery have different
    visual statistics (undamaged vs. damaged structures) that separate
    filters can specialize on.

    fusion_mode="concat" (the first version tested) concatenates the raw
    pre- and post-disaster latent maps channel-wise, leaving the classifier
    to implicitly learn "what changed" from the two raw feature sets.
    fusion_mode="post_and_diff" concatenates the post-disaster latent map
    with the elementwise absolute difference between post and pre -- an
    explicit change signal, the standard representation in the bitemporal
    change-detection literature, so the classifier is not required to
    re-derive it from raw concatenation. Zarski & Miszczak's own ablation
    likewise found plain concatenation their weakest fusion function.

    Either way, the fused representation is the same shape (channel count),
    so it feeds into the same MobileNetV2-inspired classification path used
    everywhere else in this file. Everything downstream of it (classifier
    path, geo branch, fusion head) is identical to build_full_model, reused
    as-is."""
    if image_summary_dim is None:
        image_summary_dim = IMAGE_SUMMARY_DIM if num_classes == 1 else 16

    pre_input = layers.Input(shape=image_size, name="pre_image_input")
    post_input = layers.Input(shape=image_size, name="post_image_input")

    if share_encoder:
        encoder_input, encoder_latent = build_autoencoder_encoder(image_size)
        shared_encoder = models.Model(encoder_input, encoder_latent, name="shared_bitemporal_encoder")
        pre_latent = shared_encoder(pre_input)
        post_latent = shared_encoder(post_input)
    else:
        pre_encoder_input, pre_encoder_latent = build_autoencoder_encoder(image_size)
        pre_encoder = models.Model(pre_encoder_input, pre_encoder_latent, name="pre_bitemporal_encoder")
        post_encoder_input, post_encoder_latent = build_autoencoder_encoder(image_size)
        post_encoder = models.Model(post_encoder_input, post_encoder_latent, name="post_bitemporal_encoder")
        pre_latent = pre_encoder(pre_input)
        post_latent = post_encoder(post_input)

    if fusion_mode == "concat":
        fused_latent = layers.Concatenate(axis=-1, name="bitemporal_fusion")([post_latent, pre_latent])
    elif fusion_mode == "post_and_diff":
        diff_latent = layers.Lambda(
            lambda pair: tf.abs(pair[0] - pair[1]), name="bitemporal_diff"
        )([post_latent, pre_latent])
        fused_latent = layers.Concatenate(axis=-1, name="bitemporal_fusion")([post_latent, diff_latent])
    else:
        raise ValueError(f"unknown fusion_mode: {fusion_mode!r}")
    latent_shape = fused_latent.shape[1:]

    classifier_path_input, image_summary = build_image_classifier_path(
        latent_shape, use_residual=use_residual, l2_reg=l2_reg, image_summary_dim=image_summary_dim
    )
    classifier_path_model = models.Model(classifier_path_input, image_summary, name="mobilenet_inspired_path")
    image_summary_out = classifier_path_model(fused_latent)

    geo_input, geo_embedding = build_geo_branch(geo_dim=2, embed_dim=GEO_EMBED_DIM)
    fused = layers.Concatenate(name="fusion")([image_summary_out, geo_embedding])

    fusion_dim = image_summary_dim + GEO_EMBED_DIM
    head_input, head_output = build_classifier_head(
        fusion_dim=fusion_dim, num_classes=num_classes, dropout_rate=dropout_rate, l2_reg=l2_reg,
    )
    head_model = models.Model(head_input, head_output, name="classifier_head")
    outputs = head_model(fused)

    return models.Model([pre_input, post_input, geo_input], outputs, name="autoencoder_c_mobilenetv2_bitemporal")


# Depths tapped for multi-stage fusion in build_full_model_bitemporal_pretrained,
# spanning MobileNetV2's resolution range at a 128x128 input (empirically
# confirmed shapes: 64x64x96 -> 32x32x144 -> 16x16x192 -> 8x8x576 ->
# 4x4x1280). Zarski & Miszczak (2024) fuse at multiple stages too, but with
# a from-scratch network; this fuses at multiple stages of a pretrained
# backbone, which is neither what they do (no pretraining -- their own
# paper lists it as future work) nor what this paper's earlier from-scratch
# bitemporal variant does (single fusion point, after the full encoder).
# out_relu (the final, most semantically rich stage) is included so the
# fused representation isn't missing the network's highest-level features.
MOBILENETV2_FUSION_STAGES = (
    "block_1_expand_relu",
    "block_3_expand_relu",
    "block_6_expand_relu",
    "block_13_expand_relu",
    "out_relu",
)


def build_full_model_bitemporal_pretrained(image_size=IMAGE_SIZE, num_classes=1,
                                            dropout_rate=0.3, l2_reg=1e-4, image_summary_dim=None,
                                            share_backbone=False, fine_tune_backbone=False,
                                            fusion_stages=MOBILENETV2_FUSION_STAGES):
    """Pretrained, multi-stage-fusion bitemporal variant: two ImageNet-pretrained
    MobileNetV2 backbones (one per pre-/post-disaster image), fused at several
    depths during feature extraction rather than once at the end.

    share_backbone=False (default) gives each branch its own independently-
    weighted backbone, matching the earlier from-scratch bitemporal variant's
    finding that separate branches beat one shared branch on this task.
    fine_tune_backbone=False (default) freezes the pretrained weights --
    a standard, safe first transfer-learning setting; set True to unfreeze
    and fine-tune them once a frozen run's ceiling is known.

    At each of fusion_stages' four depths, the two branches' feature maps are
    concatenated channel-wise, then globally average-pooled to a fixed-size
    vector; the four stage vectors are concatenated into one multi-scale
    summary and projected down to image_summary_dim, then fused with the
    geolocation embedding exactly as in build_full_model_bitemporal. The
    downstream fusion head (build_classifier_head) is reused unchanged."""
    if image_summary_dim is None:
        image_summary_dim = IMAGE_SUMMARY_DIM if num_classes == 1 else 16

    pre_input = layers.Input(shape=image_size, name="pre_image_input")
    post_input = layers.Input(shape=image_size, name="post_image_input")

    def _make_backbone(name):
        base = tf.keras.applications.MobileNetV2(
            input_shape=image_size, include_top=False, weights="imagenet"
        )
        base.trainable = fine_tune_backbone
        stage_outputs = [base.get_layer(stage_name).output for stage_name in fusion_stages]
        return models.Model(base.input, stage_outputs, name=f"{name}_multistage")

    if share_backbone:
        shared_backbone = _make_backbone("shared_pretrained_backbone")
        pre_stages = shared_backbone(pre_input)
        post_stages = shared_backbone(post_input)
    else:
        pre_backbone = _make_backbone("pre_pretrained_backbone")
        post_backbone = _make_backbone("post_pretrained_backbone")
        pre_stages = pre_backbone(pre_input)
        post_stages = post_backbone(post_input)

    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    stage_vectors = []
    for i, (pre_stage, post_stage) in enumerate(zip(pre_stages, post_stages)):
        fused_stage = layers.Concatenate(axis=-1, name=f"stage{i}_fusion")([post_stage, pre_stage])
        pooled = layers.GlobalAveragePooling2D(name=f"stage{i}_pool")(fused_stage)
        stage_vectors.append(pooled)

    multiscale = layers.Concatenate(name="multiscale_fusion")(stage_vectors) if len(stage_vectors) > 1 else stage_vectors[0]
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg, name="multiscale_dense1")(multiscale)
    image_summary_out = layers.Dense(
        image_summary_dim, activation="relu", kernel_regularizer=reg, name="multiscale_image_summary"
    )(x)

    geo_input, geo_embedding = build_geo_branch(geo_dim=2, embed_dim=GEO_EMBED_DIM)
    fused = layers.Concatenate(name="fusion")([image_summary_out, geo_embedding])

    fusion_dim = image_summary_dim + GEO_EMBED_DIM
    head_input, head_output = build_classifier_head(
        fusion_dim=fusion_dim, num_classes=num_classes, dropout_rate=dropout_rate, l2_reg=l2_reg,
    )
    head_model = models.Model(head_input, head_output, name="classifier_head")
    outputs = head_model(fused)

    return models.Model([pre_input, post_input, geo_input], outputs, name="pretrained_multistage_bitemporal")


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

    bitemporal_model = build_full_model_bitemporal(num_classes=4, use_residual=True)
    dummy_pre = np.random.rand(2, *IMAGE_SIZE).astype("float32")
    out_bt = bitemporal_model.predict([dummy_pre, dummy_images, dummy_geo], verbose=0)
    assert out_bt.shape == (2, 4), f"bitemporal severity output shape wrong: {out_bt.shape}"
    assert np.allclose(out_bt.sum(axis=1), 1.0, atol=1e-4), "bitemporal softmax rows don't sum to 1"
    n_params_bitemporal = bitemporal_model.count_params()
    encoder_params = models.Model(*build_autoencoder_encoder()).count_params()
    # bitemporal adds one shared encoder call (0 extra weights) plus a wider
    # first conv in the classifier path (256 fused channels in vs. 128) --
    # if the encoder were accidentally duplicated instead of shared, this
    # would be at least one more full encoder's worth of parameters higher.
    assert n_params_bitemporal < n_params_severity + 2 * encoder_params, (
        "bitemporal model is far larger than the shared-encoder design should produce -- "
        "check the encoder is being reused, not rebuilt, for the pre/post branches"
    )

    unshared_model = build_full_model_bitemporal(num_classes=4, use_residual=True, share_encoder=False)
    out_unshared = unshared_model.predict([dummy_pre, dummy_images, dummy_geo], verbose=0)
    assert out_unshared.shape == (2, 4), f"unshared bitemporal output shape wrong: {out_unshared.shape}"
    n_params_unshared = unshared_model.count_params()
    # unshared adds one full extra encoder's worth of independent weights
    # relative to the shared version -- if it didn't, the two encoders
    # would accidentally be tied together.
    assert n_params_unshared > n_params_bitemporal + encoder_params * 0.9, (
        "unshared bitemporal model isn't meaningfully larger than the shared version -- "
        "check the two encoders are actually independent, not accidentally shared"
    )

    diff_model = build_full_model_bitemporal(num_classes=4, use_residual=True, fusion_mode="post_and_diff")
    out_diff = diff_model.predict([dummy_pre, dummy_images, dummy_geo], verbose=0)
    assert out_diff.shape == (2, 4), f"post_and_diff bitemporal output shape wrong: {out_diff.shape}"
    assert np.allclose(out_diff.sum(axis=1), 1.0, atol=1e-4), "post_and_diff softmax rows don't sum to 1"
    # post_and_diff feeds the same channel width (post-latent + diff-latent,
    # both 128-channel) into the classifier path as concat (post + pre, also
    # both 128-channel) -- same total parameter count, only the content differs.
    assert diff_model.count_params() == bitemporal_model.count_params(), (
        "post_and_diff and concat fusion should produce the same parameter count "
        "(same channel width feeding the classifier path), only the fused content differs"
    )
    try:
        build_full_model_bitemporal(num_classes=4, fusion_mode="not_a_real_mode")
        raise AssertionError("build_full_model_bitemporal should reject an unknown fusion_mode")
    except ValueError:
        pass

    pretrained_model = build_full_model_bitemporal_pretrained(num_classes=4)
    out_pretrained = pretrained_model.predict([dummy_pre, dummy_images, dummy_geo], verbose=0)
    assert out_pretrained.shape == (2, 4), f"pretrained bitemporal output shape wrong: {out_pretrained.shape}"
    assert np.allclose(out_pretrained.sum(axis=1), 1.0, atol=1e-4), "pretrained bitemporal softmax rows don't sum to 1"
    n_params_pretrained = pretrained_model.count_params()
    n_trainable_pretrained = sum(int(np.prod(v.shape)) for v in pretrained_model.trainable_weights)
    # with the backbones frozen (default), only the fusion head + geo branch +
    # multiscale projection should be trainable -- a small fraction of the
    # total parameter count, since the two MobileNetV2 backbones dominate it.
    assert n_trainable_pretrained < n_params_pretrained * 0.2, (
        f"expected frozen backbones to leave only a small trainable fraction, "
        f"got {n_trainable_pretrained:,} trainable of {n_params_pretrained:,} total"
    )

    print(f"self-test OK -- latent map shape: {latent.shape}, binary params: {n_params_binary:,}, "
          f"severity params: {n_params_severity:,}, bitemporal severity params: {n_params_bitemporal:,}, "
          f"unshared bitemporal params: {n_params_unshared:,}, "
          f"pretrained multistage params: {n_params_pretrained:,} ({n_trainable_pretrained:,} trainable)")


if __name__ == "__main__":
    _self_test()
