"""
Smart NeuroCare — Advanced Medical Image Preprocessing Module

Provides clinically validated image enhancement utilities for Brain MRI:
1. Brain Auto-Cropping (`crop_brain_contour`): Strips dark borders & background noise
   using Otsu thresholding & contour extraction, focusing CNN receptive fields
   directly on neurological tissue.
2. Contrast-Limited Adaptive Histogram Equalization (CLAHE):
   Normalizes intra-scanner illumination differences (1.5T vs 3.0T MRI variance)
   and sharpens tumor boundary definition.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional


def crop_brain_contour(image: Image.Image, padding: int = 10) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Finds the extreme outer boundary of brain tissue and crops the image to eliminate
    uninformative black background space.
    
    Args:
        image: PIL Image in RGB or Grayscale.
        padding: Pixel padding to preserve surrounding skull and meninges.
        
    Returns:
        cropped_image (PIL.Image), bbox (x, y, w, h in original image coordinates)
    """
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Gaussian blur to remove high-frequency scanner noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold the image (Otsu's binarization automatically calculates optimal threshold)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing and erosion to fill internal brain cavities & isolate brain mass
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    eroded = cv2.erode(closed, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to original image if no contour detected
        w, h = image.size
        return image, (0, 0, w, h)
        
    # Get largest contour corresponding to the brain
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Add safe padding within image bounds
    img_h, img_w = img_np.shape[:2]
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img_w, x + w + padding)
    y_end = min(img_h, y + h + padding)
    
    cropped_np = img_np[y_start:y_end, x_start:x_end]
    cropped_pil = Image.fromarray(cropped_np)
    
    bbox = (x_start, y_start, x_end - x_start, y_end - y_start)
    return cropped_pil, bbox


def apply_clahe_enhancement(image: Image.Image, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Image.Image:
    """
    Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE) on the luminance channel
    (LAB color space) to enhance lesion contrast without over-amplifying background noise.
    """
    img_np = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_np)


def preprocess_mri(
    image: Image.Image,
    auto_crop: bool = True,
    enhance_contrast: bool = True
) -> Image.Image:
    """
    Complete clinical-grade MRI preprocessing pipeline.
    """
    processed = image
    if auto_crop:
        processed, _ = crop_brain_contour(processed)
    if enhance_contrast:
        processed = apply_clahe_enhancement(processed)
    return processed
