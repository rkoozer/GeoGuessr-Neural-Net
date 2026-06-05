"""
GeoGuessr Assistant — real-time country predictor
Captures your screen every N seconds while you play GeoGuessr and displays
running country predictions using the trained CNN.

Usage (Windows / Linux with display):
    python geoguessr_assistant.py

Requirements:
    pip install torch torchvision pillow mss
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import mss
import time
import os
from collections import defaultdict


# ── Model Definition (must match training code exactly) ──────────────────────
class GeoConvModelV1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 256),          nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.block4(x); x = self.block5(x)
        return self.classifier(x)


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "geo_cnn_weights_v13_min100.pth"
INTERVAL   = 3.0   # seconds between screenshots
TOP_N      = 5     # number of top countries to show
MIN_CONF   = 0.05  # minimum confidence threshold to display


# ── Load Model ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    num_classes = checkpoint["num_classes"]
    classes     = checkpoint["classes"]
    accuracy    = checkpoint.get("accuracy", None)
    model       = GeoConvModelV1(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    acc_str = f"{accuracy:.3f}" if accuracy is not None else "unknown"
    print(f"Loaded checkpoint | {num_classes} classes | accuracy: {acc_str}")
else:
    raise ValueError(
        "Unrecognized checkpoint format. Expected a dict with 'model_state_dict'. "
        "Re-save your model using the new checkpoint format in the training notebook."
    )

model.eval()


# ── Transform (must match eval_transform in training) ────────────────────────
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Assistant Loop ────────────────────────────────────────────────────────────
def run_assistant():
    vote_totals      = defaultdict(float)
    screenshot_count = 0

    print(f"\nGeoGuessr Assistant running on {device.upper()}")
    print(f"Model: {MODEL_PATH}  |  Countries: {num_classes}")
    print(f"Taking screenshot every {INTERVAL}s — Press Ctrl+C to stop\n")

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor — change index for other screens

        try:
            while True:
                # Capture screen
                raw = sct.grab(monitor)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

                # Run through model
                tensor = eval_transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0]

                # Accumulate weighted votes
                top_probs, top_idx = probs.topk(TOP_N)
                for prob, idx in zip(top_probs, top_idx):
                    vote_totals[classes[idx]] += prob.item()

                screenshot_count += 1

                # Display running results
                os.system("cls" if os.name == "nt" else "clear")
                print(f"GeoGuessr Assistant  |  Screenshots: {screenshot_count}  |  Device: {device.upper()}")
                print("=" * 65)

                sorted_votes = sorted(vote_totals.items(), key=lambda x: x[1], reverse=True)

                for country, score in sorted_votes[:TOP_N]:
                    confidence = score / screenshot_count
                    if confidence < MIN_CONF:
                        continue
                    bar    = "█" * int(confidence * 40)
                    marker = " ◄ TOP GUESS" if country == sorted_votes[0][0] else ""
                    print(f"  {country:30s}  {confidence * 100:5.1f}%  {bar}{marker}")

                print("\n[Press Ctrl+C to stop and see final prediction]")
                time.sleep(INTERVAL)

        except KeyboardInterrupt:
            print("\n" + "=" * 65)
            print("FINAL PREDICTION")
            print("=" * 65)
            sorted_votes = sorted(vote_totals.items(), key=lambda x: x[1], reverse=True)
            for i, (country, score) in enumerate(sorted_votes[:TOP_N]):
                confidence = score / screenshot_count
                print(f"  #{i + 1}  {country:30s}  {confidence * 100:.1f}%")
            print("=" * 65)


if __name__ == "__main__":
    run_assistant()
