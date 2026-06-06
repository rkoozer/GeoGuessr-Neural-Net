"""
dataset.py — data loading for the merged GeoGuessr + GSV Cities dataset.

Merges two Kaggle datasets:
  - ubitquitin/geolocation-geoguessr-images-50k  (primary)
  - amaralibey/gsv-cities                         (supplemental)

Countries with fewer than MIN_IMAGES are dropped; each is capped at MAX_IMAGES.
"""
import os
import glob
from collections import defaultdict, Counter

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


IMG_SIZE   = 224
BATCH_SIZE = 256
VALID_FRAC = 0.15
TEST_FRAC  = 0.15
MIN_IMAGES = 100
MAX_IMAGES = 2000

CITY_TO_COUNTRY = {
    'Bangkok': 'Thailand', 'Barcelona': 'Spain', 'Boston': 'United States',
    'Brussels': 'Belgium', 'BuenosAires': 'Argentina', 'Chicago': 'United States',
    'Lisbon': 'Portugal', 'London': 'United Kingdom', 'LosAngeles': 'United States',
    'Madrid': 'Spain', 'Medellin': 'Colombia', 'Melbourne': 'Australia',
    'MexicoCity': 'Mexico', 'Miami': 'United States', 'Minneapolis': 'United States',
    'OSL': 'Norway', 'Osaka': 'Japan', 'PRG': 'Czech Republic', 'PRS': 'France',
    'Phoenix': 'United States', 'Rome': 'Italy', 'TRT': 'Turkey',
    'WashingtonDC': 'United States',
}

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MergedGeoDataset(Dataset):
    """Wraps a list of (image_path, class_index) tuples with an optional transform."""
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_merged_dataset(data_root, gsv_path, random_seed=42):
    """
    Merge primary GeoGuessr-50k dataset with GSV Cities.
    Returns: all_samples, classes, country_to_idx
    """
    np.random.seed(random_seed)

    full_dataset  = datasets.ImageFolder(root=data_root, transform=None)
    all_classes   = full_dataset.classes
    label_counts  = Counter(full_dataset.targets)
    valid_classes = {idx for idx, count in label_counts.items() if count >= MIN_IMAGES}

    print(f"Primary dataset : {len(full_dataset)} images, {len(all_classes)} countries")
    print(f"After min-{MIN_IMAGES} filter: {len(valid_classes)} countries remain")

    gsv_samples = []
    for city, country in CITY_TO_COUNTRY.items():
        city_folder = os.path.join(gsv_path, city)
        if os.path.exists(city_folder):
            for img_path in glob.glob(os.path.join(city_folder, "*.jpg")):
                gsv_samples.append((img_path, country))
    np.random.shuffle(gsv_samples)
    print(f"GSV Cities      : {len(gsv_samples)} images, {len(set(c for _, c in gsv_samples))} countries")

    classes        = sorted(set(all_classes[i] for i in valid_classes) | set(CITY_TO_COUNTRY.values()))
    country_to_idx = {name: idx for idx, name in enumerate(classes)}
    print(f"Merged classes  : {len(classes)} countries")

    country_seen = defaultdict(int)
    all_samples  = []
    for img_path, label_idx in full_dataset.samples:
        country = all_classes[label_idx]
        if label_idx in valid_classes and country_seen[country] < MAX_IMAGES:
            all_samples.append((img_path, country_to_idx[country]))
            country_seen[country] += 1
    for img_path, country in gsv_samples:
        if country_seen[country] < MAX_IMAGES:
            all_samples.append((img_path, country_to_idx[country]))
            country_seen[country] += 1

    print(f"Total samples   : {len(all_samples)}")
    return all_samples, classes, country_to_idx


def make_dataloaders(all_samples, random_seed=42):
    """
    Split samples into train/val/test sets and return DataLoaders.
    Returns: train_loader, valid_loader, test_loader
    """
    np.random.seed(random_seed)
    n       = len(all_samples)
    indices = list(range(n))
    np.random.shuffle(indices)

    test_split  = int(np.floor(TEST_FRAC  * n))
    valid_split = int(np.floor(VALID_FRAC * n))
    test_idx    = indices[:test_split]
    valid_idx   = indices[test_split:test_split + valid_split]
    train_idx   = indices[test_split + valid_split:]

    train_loader = DataLoader(MergedGeoDataset([all_samples[i] for i in train_idx], train_transform),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(MergedGeoDataset([all_samples[i] for i in valid_idx], eval_transform),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(MergedGeoDataset([all_samples[i] for i in test_idx],  eval_transform),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=8, pin_memory=True, persistent_workers=True)

    print(f"Train: {len(train_idx)} | Valid: {len(valid_idx)} | Test: {len(test_idx)}")
    return train_loader, valid_loader, test_loader
