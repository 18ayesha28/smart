"""
Smart NeuroCare — Segmentation Model Training (U-Net)

Trains on the LGG MRI Segmentation dataset (image + mask .tif pairs per
patient), produced by prepare_data.py at data/segmentation/<patient_id>/.

CRITICAL: split is done by PATIENT, not by individual slice. Putting slices
from the same patient in both train and validation leaks information
(adjacent slices look almost identical) and gives a falsely high Dice score.

pip install opencv-python
"""

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2

from unet_segmentation import UNet, DiceBCELoss, dice_score

random.seed(42)
IMG_SIZE = 256


class LGGSegmentationDataset(Dataset):
    """
    Expects: data/segmentation/<patient_id>/<slice>.tif + <slice>_mask.tif
    """

    def __init__(self, root_dir: str, patient_ids: list, augment: bool = False):
        self.samples = []
        self.augment = augment
        for patient_id in patient_ids:
            patient_dir = os.path.join(root_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue
            for fname in os.listdir(patient_dir):
                if fname.endswith("_mask.tif") or not fname.endswith(".tif"):
                    continue
                mask_name = fname.replace(".tif", "_mask.tif")
                mask_path = os.path.join(patient_dir, mask_name)
                if os.path.exists(mask_path):
                    self.samples.append((os.path.join(patient_dir, fname), mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if random.random() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        image = image.astype(np.float32) / 255.0
        image_t = torch.from_numpy(image).unsqueeze(0)   # (1, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)      # (1, H, W)
        return image_t, mask_t


def get_patient_split(root_dir: str, val_fraction: float = 0.15, test_fraction: float = 0.15):
    patients = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    random.shuffle(patients)

    n_test = int(len(patients) * test_fraction)
    n_val = int(len(patients) * val_fraction)

    test_patients = patients[:n_test]
    val_patients = patients[n_test:n_test + n_val]
    train_patients = patients[n_test + n_val:]

    print(f"Patient-level split: {len(train_patients)} train / {len(val_patients)} val / {len(test_patients)} test")
    return train_patients, val_patients, test_patients


def train_segmentation_model(data_dir: str = "./data/segmentation", epochs: int = 30,
                              batch_size: int = 16, lr: float = 1e-4,
                              device: str = "cuda" if torch.cuda.is_available() else "cpu"):

    train_patients, val_patients, test_patients = get_patient_split(data_dir)

    train_ds = LGGSegmentationDataset(data_dir, train_patients, augment=True)
    val_ds = LGGSegmentationDataset(data_dir, val_patients, augment=False)
    test_ds = LGGSegmentationDataset(data_dir, test_patients, augment=False)

    print(f"Slices: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)

    best_val_dice = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        val_dice_total, n_batches = 0.0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                val_dice_total += dice_score(preds, masks).item()
                n_batches += 1
        val_dice = val_dice_total / max(n_batches, 1)

        scheduler.step(val_dice)
        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} val_dice={val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_segmentation_model.pt")

    # Held-out test evaluation
    model.load_state_dict(torch.load("best_segmentation_model.pt"))
    model.eval()
    test_dice_total, n_batches = 0.0, 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            test_dice_total += dice_score(preds, masks).item()
            n_batches += 1
    print(f"\n=== Held-out Test Dice Score: {test_dice_total / max(n_batches, 1):.4f} ===")
    print("Best checkpoint saved to best_segmentation_model.pt")

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    train_segmentation_model(device=device)
