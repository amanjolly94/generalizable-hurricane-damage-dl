# Multimodal Deep Learning for Post-Hurricane Building Damage Assessment

Code accompanying "Toward Generalizable Multimodal Deep Learning for Post-Hurricane Building Damage Assessment" (Jolly, Sharma, Pandey). Reproduces every experimental result reported in the paper: binary damage classification, cross-hurricane generalization, severity classification, statistical significance testing, and Grad-CAM explainability.

**Status**: submitted to IEEE Access.

## Results

| Model | Accuracy | Precision | Recall | F1 | Params |
|---|---|---|---|---|---|
| VGG16 | 90.5% | 89.6% | 91.2% | 90.4% | 14.72M |
| MobileNetV2 | 93.5% | 93.5% | 93.1% | 93.3% | 2.26M |
| DenseNet121 | 92.4% | 93.9% | 90.3% | 92.1% | 7.04M |
| **Proposed** | **97.1%** | **97.1%** | **97.0%** | **97.1%** | **1.29M** |

Binary classification, Hurricane Harvey dataset, n=2400. Proposed vs. each baseline significant at p < 0.001 (two-proportion z-test).

**Cross-hurricane generalization** (4 leave-one-hurricane-out folds, xBD): mean zero-shot accuracy 47.8%, mean fine-tuned accuracy 77.9% (after fine-tuning on a 20% labeled slice of the held-out event).

**Severity classification** (4-class, xBD, n=8282): 64.9% overall accuracy. F1 by class: no-damage 75.2%, minor 58.1%, major 48.6%, destroyed 43.4%.

**Grad-CAM pointing-game accuracy**: 29.5% (chance baseline 25%, not statistically significant, p=0.14).

## Architecture

A convolutional autoencoder extracts a spatial feature map from satellite imagery, which passes through a MobileNetV2-inspired classification path (Conv2D, DepthwiseConv2D, BatchNorm, a residual connection) and is fused with a small geolocation embedding built from latitude and longitude. See `code/model.py`.

## Data

Two public datasets, neither included in this repository:

- **Hurricane Harvey benchmark**: Kaggle dataset `kmader/satellite-images-of-hurricane-damage`.
- **xBD**: multi-hazard building damage dataset, used here via its four hurricane subsets (Harvey, Florence, Matthew, Michael). Sourced through the Kaggle mirror `qianlanzz/xbd-dataset`.

## Repository layout

```
code/
  model.py                  Model architecture (binary and severity heads)
  train.py                  Training loop, evaluation, focal loss, two-proportion z-test inputs
  train_cross_hurricane.py  Leave-one-hurricane-out generalization experiment
  baseline_train.py         VGG16 / MobileNetV2 / DenseNet121 baselines
  gradcam_eval.py           Grad-CAM + quantitative pointing-game evaluation
  harvey_preprocess.py      Loads the Harvey dataset (geolocation is encoded in filenames)
  xbd_preprocess.py         Crops xBD building polygons into per-building patches with labels
  make_figures.py           Regenerates the paper's result figures from recorded metrics
  make_sample_figure.py     Regenerates the dataset-sample and colorspace figures

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
