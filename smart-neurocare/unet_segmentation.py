"""
Smart NeuroCare — Tumor Segmentation Model (U-Net)

Pixel-level tumor localization from MRI slices.
Loss: Dice + BCE combo (standard for medical image segmentation).

NOTE: Reference/prototyping implementation only — not clinically validated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Standard 2D U-Net for binary tumor segmentation.
    Input:  (B, 1, H, W)  -- single-channel MRI slice
    Output: (B, 1, H, W)  -- tumor probability mask
    """

    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        ch = in_channels
        for feature in features:
            self.downs.append(DoubleConv(ch, feature))
            ch = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            concat = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat)

        return torch.sigmoid(self.final_conv(x))


class DiceBCELoss(nn.Module):
    """Combo loss: Dice (handles class imbalance) + BCE (stable gradients)."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)

        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)
        intersection = (preds_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
            preds_flat.sum() + targets_flat.sum() + self.smooth
        )
        return bce_loss + dice_loss


def dice_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6):
    """Evaluation metric: Dice Similarity Coefficient."""
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def compute_tumor_measurements(mask: torch.Tensor, pixel_spacing_mm: float, slice_thickness_mm: float = None):
    """
    Given a binary segmentation mask (single slice or stack), compute:
      - area (mm^2) per slice
      - volume (mm^3) if multiple slices provided
      - approximate max diameter (mm) via bounding box on the largest slice
    """
    mask_np = (mask > 0.5).cpu().numpy()

    if mask_np.ndim == 2:  # single slice
        pixel_count = mask_np.sum()
        area_mm2 = pixel_count * (pixel_spacing_mm ** 2)
        ys, xs = mask_np.nonzero()
        max_diameter_mm = 0.0
        if len(xs) > 0:
            width_mm = (xs.max() - xs.min()) * pixel_spacing_mm
            height_mm = (ys.max() - ys.min()) * pixel_spacing_mm
            max_diameter_mm = max(width_mm, height_mm)
        return {
            "area_mm2": float(area_mm2),
            "volume_mm3": None,
            "max_diameter_mm": float(max_diameter_mm),
        }

    elif mask_np.ndim == 3 and slice_thickness_mm:  # volume stack (slices, H, W)
        voxel_volume = (pixel_spacing_mm ** 2) * slice_thickness_mm
        total_voxels = mask_np.sum()
        volume_mm3 = total_voxels * voxel_volume

        # Largest slice for diameter estimate
        slice_sums = mask_np.reshape(mask_np.shape[0], -1).sum(axis=1)
        largest_slice_idx = slice_sums.argmax()
        largest_slice = mask_np[largest_slice_idx]
        ys, xs = largest_slice.nonzero()
        max_diameter_mm = 0.0
        if len(xs) > 0:
            width_mm = (xs.max() - xs.min()) * pixel_spacing_mm
            height_mm = (ys.max() - ys.min()) * pixel_spacing_mm
            max_diameter_mm = max(width_mm, height_mm)

        return {
            "area_mm2": None,
            "volume_mm3": float(volume_mm3),
            "max_diameter_mm": float(max_diameter_mm),
        }

    raise ValueError("Unsupported mask shape or missing slice_thickness_mm for 3D volume.")


def find_tumor_circle(mask: torch.Tensor, original_width: int, original_height: int,
                       threshold: float = 0.4, padding_factor: float = 1.3,
                       min_area_fraction: float = 0.0004):
    """
    Given a segmentation mask (H, W) from the U-Net, find the tumor region and
    return a minimum enclosing circle scaled to original image coordinates.

    Unlike a "largest single contour" approach, this treats ALL above-threshold
    pixels as one point cloud (after a morphological closing pass to merge
    fragmented predictions into coherent blobs), then fits a minimum enclosing
    circle around that whole cloud. This handles patchy / fragmented
    segmentation predictions — common with moderately-trained models — much
    more reliably than requiring one single large contour to pass a threshold.

    Args:
        mask: Raw probability mask tensor of shape (H, W) from the model.
        original_width: Width of the original uploaded image.
        original_height: Height of the original uploaded image.
        threshold: Probability threshold for binarizing the mask.
        padding_factor: Multiplier to slightly enlarge the circle for visibility.
        min_area_fraction: Minimum fraction of mask area that must be above
            threshold before we consider it a real detection (filters noise).

    Returns:
        dict with keys: center_x, center_y, radius (in original image pixels),
             plus bounding_box (x, y, w, h) in original coords.
        Returns None if no tumor region is found.
    """
    mask_np = (mask > threshold).cpu().numpy().astype(np.uint8) * 255
    mask_h, mask_w = mask_np.shape

    # Merge nearby fragmented predictions into coherent blobs before measuring
    kernel = np.ones((7, 7), np.uint8)
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    mask_np = cv2.dilate(mask_np, kernel, iterations=1)

    ys, xs = np.nonzero(mask_np)
    if len(xs) == 0:
        return None

    total_area = len(xs)
    min_area = mask_h * mask_w * min_area_fraction
    if total_area < min_area:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)
    (cx, cy), radius = cv2.minEnclosingCircle(points)

    # Scale from mask space to original image space
    scale_x = original_width / mask_w
    scale_y = original_height / mask_h

    center_x = int(cx * scale_x)
    center_y = int(cy * scale_y)
    scaled_radius = int(radius * max(scale_x, scale_y) * padding_factor)

    # Also compute bounding box in original coords
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    bbox = {
        "x": int(x * scale_x),
        "y": int(y * scale_y),
        "w": int(w * scale_x),
        "h": int(h * scale_y),
    }

    return {
        "center_x": center_x,
        "center_y": center_y,
        "radius": scaled_radius,
        "bounding_box": bbox,
    }


if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1)
    dummy_input = torch.randn(2, 1, 256, 256)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # (2, 1, 256, 256)

    dummy_target = (torch.rand(2, 1, 256, 256) > 0.9).float()
    loss_fn = DiceBCELoss()
    print("Loss:", loss_fn(output, dummy_target).item())
    print("Dice score:", dice_score(output, dummy_target).item())

    # Test find_tumor_circle on a synthetic mask with a fragmented blob
    test_mask = torch.zeros(256, 256)
    test_mask[100:115, 100:110] = 0.9
    test_mask[112:120, 108:125] = 0.85  # fragmented, offset blob nearby
    result = find_tumor_circle(test_mask, original_width=512, original_height=512)
    print("find_tumor_circle test result:", result)