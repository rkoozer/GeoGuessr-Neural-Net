# GeoGuessr Country Classifier

A CNN trained from scratch to predict which country a Google Street View image is from, with a real-time GeoGuessr assistant.

## Installation

```bash
pip install -e .
```

## Dataset

Two datasets are merged during training:

**Primary — GeoGuessr 50k (Kaggle):**
```python
import kagglehub
path = kagglehub.dataset_download("ubitquitin/geolocation-geoguessr-images-50k")
```

**Supplemental — GSV Cities (Kaggle):**
```python
path2 = kagglehub.dataset_download("amaralibey/gsv-cities")
```

The merged dataset covers **57 countries** with a minimum of 100 images per country and a cap of 2,000. GSV Cities images are mapped from city names to country labels using a built-in city→country dictionary. Total training samples after merge: ~50k.

## Project Structure

```
geoguessr-classifier/
├── notebooks/
│   ├── data_demo.ipynb              # Dataset loading demo (start here)
│   └── DSCI410_CNN_Project.ipynb    # Full training notebook (v15)
├── geoguessr/
│   ├── __init__.py
│   ├── model.py                     # GeoConvModelV1 architecture
│   ├── dataset.py                   # MergedGeoDataset, data loading
│   └── train.py                     # Training loop
├── geoguessr_assistant.py           # Real-time screen capture assistant
├── setup.py
└── README.md
```

## Usage

### Training

Open `notebooks/DSCI410_CNN_Project.ipynb` and run all cells. The notebook loads a previous checkpoint and fine-tunes from it — update `MODEL_PATH` in `train_geo_model()` to point to your latest `.pth` file.

### Real-time assistant

```bash
python geoguessr_assistant.py
```

Captures your screen every 3 seconds while you play GeoGuessr and displays running country predictions with a vote-accumulation system (predictions get more stable over time). Requires a display (Windows or Linux with `$DISPLAY` set). On a headless server, run it locally instead.

Update `MODEL_PATH` at the top of `geoguessr_assistant.py` to point to your weights file.

## Weights
   To directly download the most up to date weights to run the geoguessr_assistant.py, you can download them from Hugging Face under files and versions:
   https://huggingface.co/rkoozer/geoguessr-country-classifier

### Inference

```python
import torch
from geoguessr.model import GeoConvModelV1

checkpoint = torch.load("geo_cnn_weights_v15_min100.pth", weights_only=False)
model = GeoConvModelV1(num_classes=checkpoint["num_classes"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# checkpoint["classes"] — list of country name strings
# checkpoint["accuracy"] — test accuracy at save time
```

## Model Architecture

`GeoConvModelV1` — a custom 5-block CNN trained from scratch:

| Layer   | Channels | Operation                          |
|---------|----------|------------------------------------|
| Block 1 | 3 → 32   | Conv3×3 + BN + ReLU + MaxPool2×2  |
| Block 2 | 32 → 64  | Conv3×3 + BN + ReLU + MaxPool2×2  |
| Block 3 | 64 → 128 | Conv3×3 + BN + ReLU + MaxPool2×2  |
| Block 4 | 128 → 256| Conv3×3 + BN + ReLU + MaxPool2×2  |
| Block 5 | 256 → 512| Conv3×3 + BN + ReLU + MaxPool2×2  |
| FC      | 512×7×7 → 1024 → 256 → num_classes | Dropout(0.4) ×2 |

Input: 224×224 RGB, ImageNet normalization.

## Results

| Version | Countries | Epochs | Top-1 Acc | Notes                                        |
|---------|-----------|--------|-----------|----------------------------------------------|
| v1      | 124       | 40     | 47.1%     | Baseline, unbalanced dataset                 |
|                                  ...                                                    |
| v4      | 124       | 90     | 53.5%     | Fine-tuned from v1 (overfit)                 |
| v5      | 76        | 40     | 21.8%     | Balanced dataset experiment. Dropped countries with less than 50 images and capped outlier counties to 1000 images |
|                                  ...                                                    |
| v7      | 76        | 40     | 31.5%     | Balanced dataset converged                   |
| v8      | 77        | 40     | 36.8%     | Brought in and merged extra data             |
|                                  ...                                                    |
| v10     | 77        | 40     | 48.7%     | Merged dataset converged                     |
| v11     | 57        | 40     | 40.1%     | Continuation from merged dataset, but new min of 100 images per country and new max of 2000|
|                                  ...                                                    |
| v15     | 57        | 60     | 55.2%     | Fine-tuned from v14, reduced augmentation (no ColorJitter/RandomErasing), more consistant distances between train, test, and validation accuracy |

**Training config (v15):** lr=1e-5, Adam + weight decay 1e-4, CosineAnnealingLR, batch size 256, data augmentation (random crop, flip, rotation, random grayscale).

## Checkpoint Format

All checkpoints from v5 onward use the following format:

```python
{
    "model_state_dict": ...,
    "num_classes": 57,
    "classes": [...],        # ordered list of country name strings
    "architecture": "GeoConvModelV1",
    "accuracy": 0.500,
    "epochs_trained": 60,
}
```
