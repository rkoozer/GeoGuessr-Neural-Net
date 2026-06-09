# GeoGuessr Country Classifier

A convolutional neural network trained from scratch to predict which country a Google Street View image is from, with a real-time GeoGuessr assistant that watches your screen and predicts your location while you play.

---

## Introduction

GeoGuessr is a geography game where players are dropped into a random Google Street View location and must guess where in the world they are based solely on visual cues — road signs, vegetation, architecture, terrain, and driving-side conventions. Skilled players develop an intuition for these cues over time. This project asks: can a CNN learn those same cues from raw pixels alone?

The goal is to train an image classification model that takes a 224×224 street view image as input and outputs a probability distribution over 57 countries. Rather than using a pretrained backbone, the model is built and trained from scratch using PyTorch to better understand how convolutional networks learn geographic features. A secondary goal is to deploy the model as a real-time GeoGuessr assistant that captures your screen every few seconds and accumulates predictions to give a stable country guess over time.

This project is motivated by curiosity about what low-level visual features are sufficient for geographic localization, and whether a relatively simple CNN architecture can reach meaningful accuracy on a genuinely hard 57-class classification problem.

---

## Installation

```bash
git clone https://github.com/rkoozer/GeoGuessr-Neural-Net
cd GeoGuessr-Neural-Net
pip install -e .
```

---

## Dataset

Two Kaggle datasets are merged during training:

**Primary — GeoGuessr 50k:**
```python
import kagglehub
path = kagglehub.dataset_download("ubitquitin/geolocation-geoguessr-images-50k")
```
Contains ~50,000 Google Street View images organized into country folders across 124 countries. The raw dataset has severe class imbalance — the United States alone has over 12,000 images while many countries have fewer than 20.

**Supplemental — GSV Cities:**
```python
path2 = kagglehub.dataset_download("amaralibey/gsv-cities")
```
Contains urban street-level images organized by city. City names are mapped to country labels using a built-in dictionary covering 23 cities across 15 countries, adding coverage for underrepresented regions.

**Preprocessing and balancing:**
- Countries with fewer than 100 images total (after merging) are dropped
- Each country is capped at 2,000 images to reduce class imbalance
- Final dataset: ~50,000 images across **57 countries**
- Split: 70% train / 15% validation / 15% test (fixed random seed)

Training images use data augmentation: random crop (256→224), horizontal flip, rotation (±15°), and random grayscale. Evaluation images are resized to 224×224 with ImageNet normalization only.

**Weights on Hugging Face:**
Download the latest weights (v15) from:
https://huggingface.co/rkoozer/geoguessr-country-classifier

**Weights on Talapas:**
```
/gpfs/home/rkoozer/geo_cnn_weights_v15_min100.pth
```

---

## Project Structure

```
GeoGuessr-Neural-Net/
├── geoguessr/
│   ├── __init__.py
│   ├── model.py               # GeoConvModelV1 architecture
│   ├── dataset.py             # MergedGeoDataset, data loading, transforms
│   └── train.py               # Training loop, evaluation, top-5 accuracy
├── notebooks/
│   ├── data_demo.ipynb        # Dataset loading demo with visualizations
│   ├── evaluation.ipynb       # Model evaluation, metrics, and prediction plots
│   └── GeoGuessr_CNN_Project.ipynb  # Full training notebook (v15)
├── geoguessr_assistant.py     # Real-time screen capture assistant
├── setup.py
└── README.md
```

---

## How to Train

**Option 1 — Python script (recommended on Talapas):**
```bash
python geoguessr/train.py
```
Edit `LOAD_CHECKPOINT`, `SAVE_CHECKPOINT`, `NUM_EPOCHS`, and `LEARNING_RATE` at the top of `train.py` before running.

**Option 2 — Jupyter notebook:**
Open `notebooks/GeoGuessr_CNN_Project.ipynb` and run all cells. Datasets download automatically via `kagglehub`.

**Training config (v15):**
- Optimizer: Adam, lr=1e-5, weight decay=1e-4
- Scheduler: CosineAnnealingLR
- Batch size: 256, Epochs: 60 (fine-tuned from v14)
- Augmentation: random crop, horizontal flip, rotation, random grayscale

---

## Real-Time GeoGuessr Assistant

Run locally on Windows or Linux with a display:
```bash
python geoguessr_assistant.py
```
Captures your screen every 3 seconds, runs each screenshot through the model, and accumulates weighted votes. Predictions stabilize the longer you stay in one location. Press `Ctrl+C` for the final ranked prediction. Update `MODEL_PATH` at the top to point to your local weights file.

---

## Results

### Version History

| Version | Countries | Epochs | Top-1 Acc | Notes |
|---------|-----------|--------|-----------|-------|
| v1      | 124       | 40     | 47.1%     | Baseline, raw unbalanced dataset |
| v4      | 124       | 90     | 53.5%     | Fine-tuned from v1 (overfit) |
| v5      | 76        | 40     | 21.8%     | Balanced dataset, trained from scratch |
| v7      | 76        | 40     | 31.5%     | Balanced dataset converged |
| v8      | 77        | 40     | 36.8%     | Merged extra data |
| v10     | 77        | 40     | 48.7%     | Merged dataset converged |
| v11     | 57        | 40     | 40.1%     | New min 100 / max 2000 per country |
| v15     | 57        | 60     | 55.2%     | Fine-tuned from v14, reduced augmentation |

### Metrics

The primary metric is **top-1 accuracy** — the fraction of test images where the model's top prediction matches the true country label. A random baseline over 57 classes would achieve ~1.75%, so 55.2% represents a ~31× improvement over chance.

**Top-5 accuracy** (true label in top 5 predictions) is also tracked and is typically 10–15 points higher than top-1. This is the more practical metric for real GeoGuessr use, where players narrow down a region before committing to a country.

The training curve for v15 shows validation and test accuracy tracking closely throughout training (both ~50–55%), with training accuracy running ~7–10 points lower (~43–47%). This gap is caused by data augmentation making training harder than clean evaluation — it is not a sign of overfitting. The close alignment between val and test confirms the model generalizes well.

See `notebooks/evaluation.ipynb` for the full per-class accuracy breakdown, confusion matrix, and example predictions with confidence scores.

### Prediction Visualization

The evaluation notebook shows sample test images alongside the model's top-5 predicted countries and their confidence scores. Countries with highly distinctive visual signatures — Japan (unique road markings and signage), Russia (Cyrillic signs, vast landscapes), South Africa (distinctive road paint and vegetation) — are predicted reliably. Visually similar countries like Western European nations are frequently confused with each other, which matches human intuition about the difficulty of distinguishing them.

---

## Model Architecture

`GeoConvModelV1` — a custom 5-block CNN trained from scratch:

| Layer   | Channels                | Operation                          |
|---------|-------------------------|------------------------------------|
| Block 1 | 3 → 32                  | Conv3×3 + BN + ReLU + MaxPool2×2  |
| Block 2 | 32 → 64                 | Conv3×3 + BN + ReLU + MaxPool2×2  |
| Block 3 | 64 → 128                | Conv3×3 + BN + ReLU + MaxPool2×2  |
| Block 4 | 128 → 256               | Conv3×3 + BN + ReLU + MaxPool2×2  |
| Block 5 | 256 → 512               | Conv3×3 + BN + ReLU + MaxPool2×2  |
| FC      | 512×7×7 → 1024 → 256 → num_classes | Dropout(0.4) ×2       |

Input: 224×224 RGB, ImageNet normalization. Total parameters: ~103M.

---

## Limitations and Discussion

**What the model does well:**
Countries with distinctive visual signatures are predicted reliably. The vote-accumulation system in the assistant also means that even noisy individual predictions stabilize into correct guesses over multiple screenshots.

**What the model struggles with:**
- **Visually similar countries:** Western European countries share similar architecture, road markings, and vegetation. The model frequently confuses Germany, France, Belgium, and the Netherlands with each other.
- **Class imbalance:** Despite capping, English-speaking and Western European countries remain overrepresented, and the model is biased toward predicting them.
- **Domain mismatch:** Training images are Kaggle-scraped street view snapshots; real GeoGuessr rounds use slightly different camera perspectives. The assistant also captures the full browser window including UI elements, adding noise.
- **No spatial awareness:** The model processes each screenshot independently. It has no concept of panning, multiple viewpoints, or sequential frames — the vote system partially compensates but does not fully solve this.
- **Coverage:** 57 countries covers roughly half the world. Many countries are entirely absent from the dataset.

**Future directions:**
Fine-tuning a pretrained backbone (ResNet50, EfficientNet) rather than training from scratch would likely push accuracy significantly higher. Training on higher-resolution images could preserve fine-grained cues like license plates and road signs. Expanding to coordinate regression (latitude/longitude) rather than country classification would make the model more broadly applicable and match how GeoGuessr actually scores.

---

## Checkpoint Format

```python
{
    "model_state_dict": ...,
    "num_classes": 57,
    "classes": [...],        # ordered list of 57 country name strings
    "architecture": "GeoConvModelV1",
    "accuracy": 0.552,
    "epochs_trained": 60,
}
```
