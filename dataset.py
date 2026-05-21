import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from collections import defaultdict, Counter
import numpy as np


class RemappedSubset(torch.utils.data.Dataset):
    """
    Wraps an ImageFolder dataset, keeping only a subset of indices
    and remapping class labels to be contiguous (0, 1, 2, ...).

    This is needed after dropping underrepresented classes so that
    CrossEntropyLoss receives labels in range [0, num_classes).
    """
    def __init__(self, dataset, indices, label_map):
        self.dataset   = dataset
        self.indices   = indices
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, self.label_map[label]


def load_balanced_dataset(data_root, min_images=50, max_images=1000,
                           batch_size=256, valid_frac=0.15, test_frac=0.15,
                           num_workers=8, random_seed=42):
    """
    Loads and balances the GeoGuessr dataset.

    - Drops countries with fewer than min_images images
    - Caps countries with more than max_images images
    - Splits into train / validation / test loaders

    Returns:
        train_loader, valid_loader, test_loader, classes, num_classes
    """
    np.random.seed(random_seed)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    full_train = datasets.ImageFolder(root=data_root, transform=train_transform)
    full_eval  = datasets.ImageFolder(root=data_root, transform=eval_transform)
    all_classes = full_train.classes

    # Drop classes with too few images, cap classes with too many
    label_counts  = Counter(full_train.targets)
    valid_classes = {idx for idx, count in label_counts.items() if count >= min_images}

    class_seen       = defaultdict(int)
    balanced_indices = []
    for idx, label in enumerate(full_train.targets):
        if label in valid_classes and class_seen[label] < max_images:
            balanced_indices.append(idx)
            class_seen[label] += 1

    # Remap labels to be contiguous after dropping classes
    kept_class_indices = sorted(valid_classes)
    old_to_new  = {old: new for new, old in enumerate(kept_class_indices)}
    classes     = [all_classes[i] for i in kept_class_indices]
    num_classes = len(classes)

    train_dataset = RemappedSubset(full_train, balanced_indices, old_to_new)
    eval_dataset  = RemappedSubset(full_eval,  balanced_indices, old_to_new)

    # Split indices
    n       = len(train_dataset)
    indices = list(range(n))
    np.random.shuffle(indices)

    test_split  = int(np.floor(test_frac  * n))
    valid_split = int(np.floor(valid_frac * n))
    test_idx    = indices[:test_split]
    valid_idx   = indices[test_split:test_split + valid_split]
    train_idx   = indices[test_split + valid_split:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True)
    valid_loader = DataLoader(eval_dataset,  batch_size=batch_size,
                              sampler=SubsetRandomSampler(valid_idx),
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True)
    test_loader  = DataLoader(eval_dataset,  batch_size=batch_size,
                              sampler=SubsetRandomSampler(test_idx),
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True)

    return train_loader, valid_loader, test_loader, classes, num_classes
