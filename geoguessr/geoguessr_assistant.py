"""
GeoGuessr Real-Time Assistant
------------------------------
Run this script on your local Windows machine while playing GeoGuessr.
Every few seconds it captures your screen, runs it through the trained
CNN, and displays a running confidence ranking of predicted countries.

Usage:
    python geoguessr_assistant.py

Press Ctrl+C to stop and see the final prediction.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import mss
import time
import os
from collections import defaultdict


# ── Model Definition ──────────────────────────────────────────────────────────
class GeoConvModelV1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.block4(x); x = self.block5(x)
        return self.classifier(x)


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "geo_cnn_weights_v5_balanced.pth"  # update to latest model
INTERVAL   = 3.0   # seconds between screenshots
TOP_N      = 5     # countries to display
MIN_CONF   = 0.05  # minimum confidence to show

# Down-weight countries that were over-represented in training data
BIAS_CORRECTION = {
    "United States": 0.3,
    "Japan":         0.6,
    "France":        0.6,
    "United Kingdom": 0.7,
    "Brazil":        0.7,
    "Russia":        0.7,
    "Australia":     0.7,
    "Canada":        0.75,
}


# ── Load Model ────────────────────────────────────────────────────────────────
device     = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(MODEL_PATH, map_location=device)

num_classes = checkpoint['num_classes']
classes     = checkpoint['classes']
model       = GeoConvModelV1(num_classes=num_classes).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded {checkpoint['architecture']} | {num_classes} classes | accuracy: {checkpoint.get('accuracy', 0):.3f}")


# ── Transform ─────────────────────────────────────────────────────────────────
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
    print(f"Taking screenshot every {INTERVAL}s — Press Ctrl+C to stop\n")

    with mss.mss() as sct:
        monitor = sct.monitors[1]

        try:
            while True:
                raw = sct.grab(monitor)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

                tensor = eval_transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0]

                top_probs, top_idx = probs.topk(TOP_N)
                for prob, idx in zip(top_probs, top_idx):
                    country        = classes[idx]
                    corrected_prob = prob.item() * BIAS_CORRECTION.get(country, 1.0)
                    vote_totals[country] += corrected_prob

                screenshot_count += 1

                os.system('cls')
                print(f"GeoGuessr Assistant | Screenshots: {screenshot_count} | Device: {device.upper()}")
                print("=" * 60)

                sorted_votes = sorted(vote_totals.items(), key=lambda x: x[1], reverse=True)
                for country, score in sorted_votes[:TOP_N]:
                    confidence = score / screenshot_count
                    if confidence < MIN_CONF:
                        continue
                    bar    = chr(9608) * int(confidence * 40)
                    marker = " <- TOP GUESS" if country == sorted_votes[0][0] else ""
                    print(f"  {country:30s}  {confidence*100:5.1f}%  {bar}{marker}")

                print("\n[Press Ctrl+C to stop and see final prediction]")
                time.sleep(INTERVAL)

        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("FINAL PREDICTION:")
            print("=" * 60)
            sorted_votes = sorted(vote_totals.items(), key=lambda x: x[1], reverse=True)
            for i, (country, score) in enumerate(sorted_votes[:TOP_N]):
                confidence = score / screenshot_count
                print(f"  #{i+1}  {country:30s}  {confidence*100:.1f}%")
            print("=" * 60)


if __name__ == "__main__":
    run_assistant()
