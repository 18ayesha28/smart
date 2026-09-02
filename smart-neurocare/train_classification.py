"""
Smart NeuroCare — Tumor Type Classification Model
Multi-class: glioma / meningioma / pituitary / notumor

Reuses the same EfficientNet-B0 backbone as detection but with a 4-way
softmax head. Trained on data/classification/{train,val,test}/<class>/
(produced by prepare_data.py).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, datasets
from cnn_detection_model import train_transforms, eval_transforms

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # alphabetical, matches ImageFolder default


class TumorClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)  # raw logits, shape (B, num_classes)


def train_classification_model(data_dir: str, epochs: int = 20, batch_size: int = 32,
                                lr: float = 1e-4,
                                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_transforms)
    val_ds = datasets.ImageFolder(f"{data_dir}/val", transform=eval_transforms)

    print("Class-to-index mapping:", train_ds.class_to_idx)
    print("Train samples per class:")
    for cls, idx in train_ds.class_to_idx.items():
        count = sum(1 for _, label in train_ds.samples if label == idx)
        print(f"  {cls}: {count}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = TumorClassificationModel(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
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

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_ds)
        accuracy = correct / total
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_classification_model.pt")

    return model, train_ds.classes


def evaluate_per_class(model, data_dir, classes, device="cuda" if torch.cuda.is_available() else "cpu"):
    test_ds = datasets.ImageFolder(f"{data_dir}/test", transform=eval_transforms)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model.eval()
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1

    print("\n=== Held-out Test Confusion Matrix (rows=actual, cols=predicted) ===")
    header = "        " + " ".join(f"{c[:8]:>8}" for c in classes)
    print(header)
    for i, cls in enumerate(classes):
        row = " ".join(f"{confusion[i, j].item():>8}" for j in range(len(classes)))
        print(f"{cls[:8]:>8} {row}")

    per_class_recall = confusion.diag().float() / confusion.sum(dim=1).clamp(min=1).float()
    for i, cls in enumerate(classes):
        print(f"Recall for {cls}: {per_class_recall[i]:.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    model, classes = train_classification_model(data_dir="./data/classification", epochs=20, device=device)
    evaluate_per_class(model, "./data/classification", classes, device=device)
    print("\nBest checkpoint saved to best_classification_model.pt")
