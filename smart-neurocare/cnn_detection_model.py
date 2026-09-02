"""
Smart NeuroCare — Brain Tumor Detection Model
Binary classification (tumor / no tumor) using transfer learning.

Framework: PyTorch
Backbone: EfficientNet-B0 (pretrained on ImageNet), fine-tuned on brain MRI.

NOTE: This is a starter/reference implementation for prototyping.
It is NOT validated for clinical use. Any real deployment requires
proper clinical validation, regulatory review, and a licensed
radiologist in the review loop.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BrainMRIDataset(Dataset):
    """
    Expects a directory structure:
        root/
          tumor/*.png
          no_tumor/*.png
    """

    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label, folder in enumerate(["no_tumor", "tumor"]):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                self.samples.append((os.path.join(folder_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Preprocessing / augmentation
# ---------------------------------------------------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                          std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TumorDetectionModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1),  # binary logit
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x).squeeze(1)  # raw logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(data_dir: str, epochs: int = 15, batch_size: int = 32, lr: float = 1e-4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

    train_ds = BrainMRIDataset(os.path.join(data_dir, "train"), transform=train_transforms)
    val_ds = BrainMRIDataset(os.path.join(data_dir, "val"), transform=eval_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = TumorDetectionModel(pretrained=True).to(device)

    # Class weighting to favor recall (missing a tumor is worse than a false alarm)
    pos_weight = torch.tensor([1.5]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()

        val_loss /= len(val_ds)
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        accuracy = correct / total

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} acc={accuracy:.4f} "
              f"recall={recall:.4f} precision={precision:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_detection_model.pt")

    return model


def predict(model: TumorDetectionModel, image_path: str,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Run inference on a single MRI image."""
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()
    return {
        "tumor_detected": prob > 0.5,
        "confidence": round(prob, 4),
    }


if __name__ == "__main__":
    # Expects data/detection/train/{tumor,no_tumor} and data/detection/val/{tumor,no_tumor}
    # (produced by prepare_data.py)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    trained_model = train_model(data_dir="./data/detection", epochs=20, device=device)

    # Held-out evaluation on data/detection/test
    test_ds = BrainMRIDataset("./data/detection/test", transform=eval_transforms)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    trained_model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = (torch.sigmoid(trained_model(images)) > 0.5).float()
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print("\n=== Held-out Test Set Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}   <-- most important: missed tumors are costly")
    print(f"F1 score:  {f1:.4f}")
    print(f"Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    print("\nBest checkpoint saved to best_detection_model.pt")
