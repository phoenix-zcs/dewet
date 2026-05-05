# dewet 🧹

Image watermark removal tool — 图片去水印工具。

Built with OpenCV and NumPy. Fast, no GPU required.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from dewet import load_image, remove_watermark, create_mask

img = load_image("watermarked.jpg")
mask = create_mask(img, [(200, 100, 150, 50)])  # (x, y, w, h)
result = remove_watermark(img, mask)
```

**Auto text watermark removal** (one-liner):

```python
from dewet import remove_text_watermark

result = remove_text_watermark("watermarked.jpg")
```

## CLI Quick Start

```bash
python examples/demo.py your_image.jpg
```

## Methods

| Method | Best For |
|--------|----------|
| `remove_watermark` | Known mask region (manual selection) |
| `remove_text_watermark` | Semi-transparent text overlays |
| `auto_inpaint` | Logo watermarks on flat backgrounds |

## License

MIT
