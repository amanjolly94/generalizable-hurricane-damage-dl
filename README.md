# Multimodal Deep Learning for Post-Hurricane Building Damage Assessment

Code accompanying "Multimodal Post-Hurricane Damage Assessment: A Head-to-Head Bitemporal Fusion Comparison" (Jolly, Sharma, Pandey, et al.). Reproduces every experimental result reported in the paper: binary damage classification, cross-hurricane generalization, severity classification, statistical significance testing, Grad-CAM explainability, and a bitemporal head-to-head comparison against a published IEEE Access architecture.

**Status**: submitted to IEEE Access.

## Architecture

A convolutional autoencoder extracts a spatial feature map from satellite imagery, which passes through a MobileNetV2-inspired classification path (Conv2D, DepthwiseConv2D, BatchNorm, a residual connection) and is fused with a small geolocation embedding built from latitude and longitude. See `code/model.py`.

For the bitemporal head-to-head comparison, a second image branch (the pre-disaster image alongside the post-disaster one) is added, keeping the same geolocation branch as everywhere else in this repo. Two variants: a lightweight from-scratch dual encoder (`build_full_model_bitemporal`), and a pretrained, multi-stage-fusion variant using two ImageNet-pretrained MobileNetV2 backbones fused at five depths (`build_full_model_bitemporal_pretrained`). See `code/model.py` and `code/train_bitemporal_xview2.py`.

## Data

Two public data sources, neither included in this repository, used in three configurations:

- **Hurricane Harvey benchmark**: Kaggle dataset `kmader/satellite-images-of-hurricane-damage`.
- **xBD**, four-hurricane subset (Harvey, Florence, Matthew, Michael), used for the cross-hurricane and severity experiments. Sourced through the Kaggle mirror `qianlanzz/xbd-dataset`.
- **xBD/xView2**, full seventeen-disaster set, reprocessed with paired pre- and post-disaster images for the bitemporal head-to-head comparison against Żarski and Miszczak (2024, IEEE Access). Same Kaggle mirror, `--all-disasters --include-pre-disaster`.

## Repository layout

```
code/
  model.py                     Model architecture (binary, severity, and bitemporal heads)
  train.py                     Training loop, evaluation, focal loss, two-proportion z-test inputs
  train_cross_hurricane.py     Leave-one-hurricane-out generalization experiment
  train_bitemporal_xview2.py   Bitemporal head-to-head comparison, both model variants
  baseline_train.py            VGG16 / MobileNetV2 / DenseNet121 baselines
  gradcam_eval.py              Grad-CAM + quantitative pointing-game evaluation
  harvey_preprocess.py         Loads the Harvey dataset (geolocation is encoded in filenames)
  xbd_preprocess.py            Crops xBD building polygons into per-building patches with labels;
                                --all-disasters and --include-pre-disaster extend this to the full
                                bitemporal xView2 set
  make_figures.py              Regenerates the paper's result figures from recorded metrics
  make_sample_figure.py        Regenerates the dataset-sample and colorspace figures

kernels/
  One folder per experiment, each a self-contained Kaggle kernel push:
  a `kernel-metadata.json` (dataset/kernel sources, GPU flag) and a driver
  script that locates the mounted data/code and calls into code/.
```

## Running an experiment

Every experiment in this paper was run on Kaggle, not locally. The `code/` modules are packaged as a Kaggle Dataset that each kernel imports at runtime; the `kernels/*/kernel-metadata.json` files declare that dependency plus the raw-data `dataset_sources` / `kernel_sources`.

To reproduce:

1. Upload the contents of `code/` as a Kaggle Dataset.
2. For each experiment folder under `kernels/`, update `kernel-metadata.json`'s `dataset_sources` to point at your own code dataset and the relevant raw-data dataset, then push it:
   ```
   kaggle kernels push -p kernels/<experiment_name>
   ```
3. Each driver script (`run_exp_*.py`) prints the `/kaggle/input` tree it finds before running, and falls back through a short list of candidate mount paths, since Kaggle's mount path differs between a `dataset_sources` entry and a `kernel_sources` entry (`/kaggle/input/datasets/<owner>/<slug>/...` vs. `/kaggle/input/notebooks/<owner>/<slug>/...`).

Results are written as `results.json` (or `gradcam_results.json`) to each kernel's `/kaggle/working` output.
