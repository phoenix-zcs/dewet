"""
dewet Web UI - 图片去水印网页版

运行: python web.py
打开浏览器访问: http://localhost:7860
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path

from dewet import remove_watermark, remove_text_watermark


def process_brush_mode(image: np.ndarray, mask_image: np.ndarray, method: str, radius: int, brush_size: int):
    """Process image using brush-drawn mask."""
    if image is None:
        return None, None

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Build mask from brush overlay
    if mask_image is not None:
        # mask_image is RGBA or RGB with the brush strokes
        if mask_image.shape[-1] == 4:
            # Use alpha channel as mask
            mask = mask_image[:, :, 3]
        else:
            # Convert to grayscale and threshold
            gray = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    else:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

    if np.sum(mask) == 0:
        # Nothing painted, return original
        return image, None

    result = remove_watermark(img, mask, method=method, radius=radius)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result, None


def process_auto_mode(image: np.ndarray, method: str, radius: int, contour_area: int, block_size: int):
    """Process image using auto text detection."""
    if image is None:
        return None, None

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = remove_text_watermark(
        img,
        method=method,
        radius=radius,
        min_contour_area=contour_area,
        adaptive_blocksize=block_size,
    )
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result, None


def save_result(result):
    """Save result and return file path."""
    if result is None:
        return None
    out_path = Path("output") / "result.png"
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return str(out_path)


with gr.Blocks(title="dewet - 图片去水印", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🧹 dewet - 图片去水印工具")
    gr.Markdown("上传带水印的图片，用画笔涂抹水印区域，点击运行即可。")

    mode = gr.Radio(
        choices=["brush", "auto_text"],
        value="brush",
        label="去水印方式",
    )
    gr.Markdown("- **画笔涂抹**（推荐）：直接在图片上涂选水印区域")
    gr.Markdown("- **自动检测**：自动识别文字水印并去除")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                label="上传图片并涂抹水印区域",
                type="numpy",
                height=450,
                tool="sketch",
                brush=gr.Brush(colors=["#FF0000"], default_size=15),
            )

            with gr.Group(visible=True) as brush_group:
                brush_size = gr.Slider(5, 80, value=15, step=1, label="画笔大小（涂大一点效果更好）")
                gr.Markdown("💡 用鼠标在图片上涂抹水印区域，红色区域就是会被去除的部分")
                gr.Markdown("💡 涂抹范围可以比水印大一点，效果会更好")

            with gr.Group(visible=False) as auto_group:
                contour_area = gr.Slider(10, 500, value=50, step=10, label="最小文字面积")
                block_size = gr.Slider(3, 51, value=31, step=2, label="检测块大小")

            with gr.Group():
                inpaint_method = gr.Radio(
                    choices=["telea", "ns"],
                    value="telea",
                    label="填充算法",
                )
                radius = gr.Slider(1, 15, value=5, step=1, label="填充半径（越大越平滑，但也越模糊）")

            btn = gr.Button("🚀 开始去水印", variant="primary", size="lg")

        with gr.Column():
            output_img = gr.Image(label="去水印结果", type="numpy", height=450)
            download_btn = gr.DownloadButton(label="💾 下载结果", value=None)

    def toggle_mode(m):
        if m == "brush":
            return {brush_group: gr.Group(visible=True), auto_group: gr.Group(visible=False)}
        else:
            return {brush_group: gr.Group(visible=False), auto_group: gr.Group(visible=True)}

    mode.change(toggle_mode, inputs=mode, outputs=[brush_group, auto_group])

    def run_process(image, mask_image, m, method, radius, bsize, area, block):
        if m == "brush":
            result, _ = process_brush_mode(image, mask_image, method, radius, bsize)
        else:
            result, _ = process_auto_mode(image, method, radius, area, block)
        out = save_result(result)
        return result, out

    btn.click(
        run_process,
        inputs=[input_img, input_img, mode, inpaint_method, radius, brush_size, contour_area, block_size],
        outputs=[output_img, download_btn],
    )

    gr.Markdown("---")
    gr.Markdown("💡 画笔涂抹模式效果最好，涂选范围稍微大一点，去水印效果更干净。")
    gr.Markdown("GitHub: https://github.com/phoenix-zcs/dewet")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
