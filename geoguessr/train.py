"""
train.py — training loop for GeoConvModelV1
"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm import tqdm
import kagglehub

from model import GeoConvModelV1
from dataset import build_merged_dataset, make_dataloaders


def train_step(model, train_loader, loss_fn, optimizer, device):
    """Run one epoch of training. Returns: (avg_loss, accuracy)"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss   = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (torch.argmax(y_pred, dim=1) == y_batch).sum().item()
        total      += y_batch.size(0)
    return total_loss / len(train_loader), correct / total


def evaluation_step(model, data_loader, loss_fn, device):
    """Evaluate model on a data loader. Returns: (avg_loss, accuracy)"""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss   = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            correct    += (torch.argmax(y_pred, dim=1) == y_batch).sum().item()
            total      += y_batch.size(0)
    return total_loss / len(data_loader), correct / total


def top5_accuracy(model, loader, device):
    """Compute top-5 accuracy — correct if true label is in top 5 predictions."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            top5 = torch.topk(model(X), 5, dim=1).indices
            correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total   += y.size(0)
    return correct / total


def train_geo_model(train_loader, valid_loader, test_loader, num_classes,
                    load_from="geo_cnn_weights_v14_min100.pth",
                    save_to="geo_cnn_weights_v15_min100.pth",
                    num_epochs=60, learning_rate=1e-5, random_seed=42):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    torch.manual_seed(random_seed)

    model = GeoConvModelV1(num_classes=num_classes).to(device)
    if load_from and os.path.exists(load_from):
        checkpoint = torch.load(load_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded {load_from} | prev accuracy: {checkpoint.get('accuracy', 'unknown')}")
    else:
        print("Training from scratch")

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []
    test_losses,  test_accs  = [], []

    print(f"\nTraining | epochs={num_epochs} | lr={learning_rate} | weight_decay=1e-4\n")
    start = timer()

    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        valid_loss, valid_acc = evaluation_step(model, valid_loader, loss_fn, device)
        test_loss,  test_acc  = evaluation_step(model, test_loader,  loss_fn, device)
        scheduler.step()

        train_losses.append(train_loss); train_accs.append(train_acc)
        valid_losses.append(valid_loss); valid_accs.append(valid_acc)
        test_losses.append(test_loss);   test_accs.append(test_acc)

    print(f"\nDone in {(timer() - start) / 60:.1f} min")
    print(f"Final test accuracy: {test_accs[-1]:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes':      num_classes,
        'classes':          classes,
        'architecture':     'GeoConvModelV1',
        'accuracy':         test_accs[-1],
        'epochs_trained':   num_epochs,
    }, save_to)
    print(f"Saved to {save_to}")

    return model, train_losses, train_accs, valid_losses, valid_accs, test_losses, test_accs


if __name__ == "__main__":
    geo_path = kagglehub.dataset_download("ubitquitin/geolocation-geoguessr-images-50k")
    gsv_path = kagglehub.dataset_download("amaralibey/gsv-cities")

    data_root = os.path.join(geo_path, "compressed_dataset")
    gsv_images_path = os.path.join(gsv_path, "Images")

    all_samples, classes, _ = build_merged_dataset(data_root, gsv_images_path)
    num_classes = len(classes)
    train_loader, valid_loader, test_loader = make_dataloaders(all_samples)

    train_geo_model(train_loader, valid_loader, test_loader, num_classes)
