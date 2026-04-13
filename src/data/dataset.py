import os
import glob
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import cv2

class BUSIDataset(Dataset):
    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

def prepare_data_loaders(data_dir: str, batch_size: int, splits: List[float], transforms: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    classes = ['benign', 'malignant']
    all_files, all_labels = [], []
    
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        # Filter out mask files
        files = [f for f in glob.glob(os.path.join(class_dir, "*.png")) if "_mask" not in f]
        all_files.extend(files)
        all_labels.extend([label_idx] * len(files))
        
    # Stratified Split (70/15/15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_files, all_labels, test_size=splits[2], stratify=all_labels, random_state=42
    )
    val_ratio = splits[1] / (splits[0] + splits[1])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
    )

    # Weighted Random Sampler
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = np.array([class_weights[t] for t in y_train])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Datasets & Loaders
    train_ds = BUSIDataset(X_train, y_train, transform=transforms['train'])
    val_ds = BUSIDataset(X_val, y_val, transform=transforms['val'])
    test_ds = BUSIDataset(X_test, y_test, transform=transforms['val'])

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader