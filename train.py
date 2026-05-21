import torch
import torch.nn as nn


def train_step(model, train_loader, loss_fn, optimizer, device):
    """
    Run one epoch of training.
    Returns: (avg_loss, accuracy)
    """
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
    """
    Evaluate model on a data loader.
    Returns: (avg_loss, accuracy)
    """
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
    """
    Compute top-5 accuracy — correct if true label is in top 5 predictions.
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            top5 = torch.topk(model(X), 5, dim=1).indices
            correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total   += y.size(0)

    return correct / total
