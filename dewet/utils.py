"""Utility functions for loading, saving, and displaying images."""

import cv2
import numpy as np
from pathlib import Path


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk (BGR format).

    Args:
        path: Path to image file (jpg, png, etc.).

    Returns:
        BGR image array (H, W, 3).

    Raises:
        FileNotFoundError: If the file doesn't exist or can't be read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save a BGR image to disk.

    Args:
        path: Output file path (extension determines format).
        image: BGR image array.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def show_image(image: np.ndarray, title: str = "dewet", scale: float = 1.0) -> None:
    """Display an image in a window (blocking).

    Args:
        image: BGR image array.
        title: Window title.
        scale: Display scale factor (0.5 = half size).
    """
    if scale != 1.0:
        h, w = image.shape[:2]
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_mask(image: np.ndarray, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Create a binary mask from rectangle regions.

    Args:
        image: Reference image (used for dimensions).
        rects: List of (x, y, w, h) rectangles.

    Returns:
        Binary mask (H, W) with white = selected regions.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for x, y, w, h in rects:
        mask[y : y + h, x : x + w] = 255
    return mask
