"""Example usage of dewet."""

import cv2
import numpy as np
from pathlib import Path

from dewet import (
    load_image,
    save_image,
    show_image,
    remove_watermark,
    remove_text_watermark,
    create_mask,
)


def demo_manual_mask(input_path: str, output_dir: str = "output"):
    """Demo 1: Manual mask inpainting."""
    img = load_image(input_path)
    h, w = img.shape[:2]

    # Manually define a watermark region (x, y, w, h)
    rects = [(w // 2 - 100, h // 2 - 30, 200, 60)]
    mask = create_mask(img, rects)

    result = remove_watermark(img, mask, method="telea", radius=5)

    out = Path(output_dir) / "manual_result.jpg"
    save_image(str(out), result)
    print(f"Saved: {out}")
    return result


def demo_auto_text(input_path: str, output_dir: str = "output"):
    """Demo 2: Automatic text watermark removal."""
    img = load_image(input_path)

    result = remove_text_watermark(
        img,
        method="telea",
        radius=3,
        adaptive_blocksize=31,
        min_contour_area=50,
    )

    out = Path(output_dir) / "auto_text_result.jpg"
    save_image(str(out), result)
    print(f"Saved: {out}")
    return result


def demo_before_after(input_path: str, output_dir: str = "output"):
    """Demo 3: Side-by-side before/after comparison."""
    img = load_image(input_path)
    h, w = img.shape[:2]

    # Guess text watermark region (center 30% of the image)
    cx, cy = w // 2, h // 2
    rects = [(cx - w // 6, cy - 30, w // 3, 60)]
    mask = create_mask(img, rects)

    cleaned = remove_watermark(img, mask, method="telea", radius=5)

    # Stack horizontally
    comparison = np.hstack((img, cleaned))

    # Draw divider
    cv2.line(comparison, (w, 0), (w, h), (0, 255, 0), 2)
    cv2.putText(comparison, "Before", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(comparison, "After", (w + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    out = Path(output_dir) / "comparison.jpg"
    save_image(str(out), comparison)
    print(f"Saved: {out}")
    return comparison


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python demo.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("=== Manual Mask Inpainting ===")
    demo_manual_mask(image_path)

    print("\n=== Auto Text Watermark Removal ===")
    demo_auto_text(image_path)

    print("\n=== Before/After Comparison ===")
    demo_before_after(image_path)
