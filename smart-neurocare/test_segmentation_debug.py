"""
Smart NeuroCare — Segmentation model diagnostic script.

Tests the trained segmentation model on TWO images:
  1. A real image FROM the LGG training data (in-distribution) — this tells us
     whether the checkpoint loaded correctly and the model works AT ALL.
  2. Your own uploaded MRI image (likely a different dataset / distribution)
     — for direct comparison.

This isolates whether "no red circle" is a model-domain-mismatch issue
(model works, just doesn't generalize to this image style) vs. an actual
bug (checkpoint didn't load, preprocessing mismatch, etc).

Run:
    python test_segmentation_debug.py
"""

import os
import glob
import numpy as np
import torch
import cv2
from PIL import Image
from unet_segmentation import UNet

IMG_SIZE = 256


def load_model():
    model = UNet(in_channels=1, out_channels=1)
    state_dict = torch.load("best_segmentation_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Sanity check: print a couple of weight stats so we can confirm this
    # isn't accidentally an untrained/randomly-initialized model
    first_conv_weight = state_dict[list(state_dict.keys())[0]]
    print(f"Checkpoint sanity check — first layer weight stats: "
          f"mean={first_conv_weight.mean().item():.6f}, std={first_conv_weight.std().item():.6f}")
    print("(if std is ~0.02-ish AND this matches a fresh random init, that could indicate "
          "the checkpoint didn't actually load — but usually trained weights look different "
          "from init, so this is just a rough smell test)\n")

    return model


def preprocess_cv2(path):
    """Matches train_segmentation.py's LGGSegmentationDataset preprocessing exactly."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0)


def preprocess_pil(path):
    """Matches app.py's inference preprocessing exactly."""
    image = Image.open(path).convert("RGB")
    gray = image.convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(gray).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def run_and_report(model, tensor, label, save_heatmap_path=None):
    with torch.no_grad():
        mask = model(tensor)[0, 0]
    n_above_04 = int((mask > 0.4).sum().item())
    n_above_02 = int((mask > 0.2).sum().item())
    print(f"[{label}]")
    print(f"  min={mask.min().item():.4f}  max={mask.max().item():.4f}  mean={mask.mean().item():.4f}")
    print(f"  pixels > 0.4: {n_above_04}   pixels > 0.2: {n_above_02}  (out of {IMG_SIZE*IMG_SIZE} total)")

    if save_heatmap_path:
        heatmap = (mask.numpy() * 255).astype(np.uint8)
        cv2.imwrite(save_heatmap_path, heatmap)
        print(f"  saved heatmap to {save_heatmap_path}")
    print()
    return mask


if __name__ == "__main__":
    if not os.path.exists("best_segmentation_model.pt"):
        print("ERROR: best_segmentation_model.pt not found in this folder.")
        exit(1)

    model = load_model()

    # 1) Test on a REAL LGG training image (in-distribution — what the model actually learned from)
    all_tifs = glob.glob("data/segmentation/*/*.tif")
    lgg_images_with_tumor = []
    for p in all_tifs:
        if "_mask" in p:
            continue
        mask_path = p.replace(".tif", "_mask.tif")
        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None and (m > 127).sum() > 50:  # has a real tumor in ground truth
                lgg_images_with_tumor.append(p)

    if lgg_images_with_tumor:
        sample_path = lgg_images_with_tumor[0]
        print(f"Testing on a REAL LGG image that HAS a ground-truth tumor: {sample_path}\n")
        tensor = preprocess_cv2(sample_path)
        run_and_report(model, tensor, "LGG in-distribution image (known to contain a tumor)",
                        save_heatmap_path="debug_lgg_heatmap.png")
    else:
        print("Could not find an LGG image with a real tumor mask under data/segmentation/ "
              "-- skipping in-distribution test. Make sure you're running this from your "
              "project folder where data/segmentation/ exists.\n")

    # 2) Test on your uploaded image -- EDIT THIS PATH
    your_image_path = r"C:/Users/Admin/Desktop/Tr-gl_679.jpg"

    if os.path.exists(your_image_path):
        tensor2 = preprocess_pil(your_image_path)
        run_and_report(model, tensor2, "Your uploaded test image",
                        save_heatmap_path="debug_your_image_heatmap.png")
    else:
        print(f"Edit 'your_image_path' at the bottom of this script to point at the exact "
              f"image file you uploaded in the app, then re-run.")

    print("=" * 70)
    print("HOW TO READ THIS:")
    print("- If the LGG image shows max > 0.4 with many pixels above threshold,")
    print("  but your uploaded image shows near-zero everywhere -> DOMAIN MISMATCH.")
    print("  The model works, it just wasn't trained on images that look like yours.")
    print("- If BOTH images show near-zero everywhere -> something is actually wrong")
    print("  with the checkpoint/model, not just generalization. Paste this output back.")
    print("=" * 70)