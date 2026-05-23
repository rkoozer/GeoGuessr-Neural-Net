# GeoGuessr Country Classifier

A CNN trained from scratch to predict which country a GeoGuessr image is from, with a real-time GeoGuessr assistant.

## Installation

```bash
pip install -e .
```

## Dataset

Downloaded automatically via kagglehub:

```python
import kagglehub
path = kagglehub.dataset_download("ubitquitin/geolocation-geoguessr-images-50k")
```

~50k street view images across 124 countries. After balancing (min 50, max 1000 per country): 76 countries, ~38k images.

## Project Structure

```
geoguessr-classifier/
├── notebooks/
│   ├── data_demo.ipynb          # Dataset loading demo (start here)
│   └── DSCI410_CNN_Project.ipynb # Full training notebook
├── geoguessr/
│   ├── __init__.py
│   ├── model.py                 # GeoConvModelV1 architecture
│   ├── dataset.py               # RemappedSubset, data loading
│   └── train.py                 # Training loop
├── geoguessr_assistant.py       # Real-time screen capture assistant
├── setup.py
└── README.md
```

## Usage

### Training
Open `notebooks/DSCI410_CNN_Project.ipynb` and run all cells.

### Real-time assistant (Windows)
```bash
python geoguessr_assistant.py
```
Captures your screen every 3 seconds while you play GeoGuessr and displays running country predictions.

### Inference
```python
from geoguessr.model import GeoConvModelV1
import torch

checkpoint = torch.load("geo_cnn_weights_v5_balanced.pth")
model = GeoConvModelV1(num_classes=checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state_dict'])
```

## Results

| Version | Countries | Epochs | Top-1 Acc | Notes |
|---------|-----------|--------|-----------|-------|
| v1 | 124 | 40 | 47.1% | Unbalanced dataset |
| v4 | 124 | 90 | 53.5% | Fine-tuned from v1 |
| v6 | 76 | 40 | 28.3% | Balanced dataset, training in progress |
