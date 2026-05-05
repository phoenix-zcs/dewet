"""
Core watermark removal algorithms.

Supports three strategies:
  1. Manual mask inpainting — user provides a mask, we fill via OpenCV inpainting
  2. Text watermark detection + removal — find text-like regions and inpaint them
  3. Auto patch-fill — clone stamp style, sample from nearby clean areas
"""

import cv2
import numpy as np


def remove_watermark(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    method: str = "telea",
    radius: int = 3,
) -> np.ndarray:
    """Remove watermark by inpainting masked regions.

    Args:
        image: Input BGR image (H, W, 3).
        mask: Binary mask (H, W) — white = region to inpaint.
              If None, returns original image.
        method: 'telea' (Navier-Stokes, edge-aware) or 'ns' (fast).
        radius: Inpainting radius (higher = smoother but slower).

    Returns:
        Inpainted BGR image.
    """
    if mask is None:
        return image.copy()

    flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(image, mask, radius, flags)


def remove_text_watermark(
    image: np.ndarray,
    method: str = "telea",
    radius: int = 3,
    adaptive_blocksize: int = 31,
    morph_kernel: int = 5,
    min_contour_area: int = 50,
) -> np.ndarray:
    """Automatically detect and remove text-like watermarks.

    Uses adaptive thresholding + morphological ops to locate
    text regions, then inpaints them.

    Args:
        image: Input BGR image (H, W, 3).
        method: Inpainting method ('telea' or 'ns').
        radius: Inpainting radius.
        adaptive_blocksize: Block size for adaptive threshold (odd).
        morph_kernel: Kernel size for morphological closing.
        min_contour_area: Minimum contour area to treat as watermark.

    Returns:
        De-watermarked BGR image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to isolate text-like regions
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, adaptive_blocksize, 2,
    )

    # Morphological closing to connect fragmented text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Filter small noise contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_contour_area:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    # Dilate mask slightly for cleaner inpaint
    mask = cv2.dilate(mask, kernel, iterations=1)

    return remove_watermark(image, mask, method, radius)


def auto_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 5,
    sample_radius: int = 50,
) -> np.ndarray:
    """Patch-based watermark removal (clone-stamp style).

    For each masked pixel, samples from the nearest clean area
    and blends. Good for logo watermarks on uniform backgrounds.

    Args:
        image: Input BGR image (H, W, 3).
        mask: Binary mask (H, W) — white = watermark region.
        patch_size: Half-size of the sample patch.
        sample_radius: Max distance to look for clean pixels.

    Returns:
        Patched BGR image.
    """
    result = image.copy()
    rows, cols = np.where(mask == 255)
    h, w = image.shape[:2]

    for y, x in zip(rows, cols):
        # Search outward from the pixel for the nearest clean pixel
        found = False
        for r in range(1, sample_radius):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 0:
                        # Sample a patch around this clean pixel
                        patch_y = max(0, ny - patch_size)
                        patch_y2 = min(h, ny + patch_size + 1)
                        patch_x = max(0, nx - patch_size)
                        patch_x2 = min(w, nx + patch_size + 1)
                        patch = result[patch_y:patch_y2, patch_x:patch_x2]
                        if patch.size > 0:
                            result[y, x] = patch[patch.shape[0] // 2, patch.shape[1] // 2]
                        found = True
                        break
                if found:
                    break
            if found:
                break

    return result
