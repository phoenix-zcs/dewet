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


def process_image(
    image: np.ndarray,
    mode: str,
    inpaint_method: str,
    radius: int,
    contour_area: int,
    block_size: int,
    # Manual mode coordinates
    x1: int,
    y1: int,
    x2: int,
    y2: int,
):
    """Process image based on selected mode."""
    if image is None:
        return None

    # Gradio gives RGB, convert to BGR for OpenCV
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if mode == "auto_text":
        # Auto detect and remove text watermark
        result = remove_text_watermark(
            img,
            method=inpaint_method,
            radius=radius,
            min_contour_area=contour_area,
            adaptive_blocksize=block_size,
        )
    elif mode == "manual_rect":
        # Manual rectangle region inpainting
        h, w = img.shape[:2]
        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        rw, rh = x2 - x1, y2 - y1

        if rw <= 0 or rh <= 0:
            result = img.copy()
        else:
            from dewet.utils import create_mask
            mask = create_mask(img, [(x1, y1, rw, rh)])
            result = remove_watermark(img, mask, method=inpaint_method, radius=radius)
    else:
        result = img.copy()

    # Convert back to RGB for Gradio display
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


with gr.Blocks(title="dewet - 图片去水印", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🧹 dewet - 图片去水印工具")
    gr.Markdown("上传带水印的图片，选择去水印方式，点击运行即可。")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="上传图片", type="numpy", height=400)

            mode = gr.Radio(
                choices=["auto_text", "manual_rect"],
                value="auto_text",
                label="去水印方式",
            )
            gr.Markdown("- **auto_text**: 自动检测文字水印并去除")
            gr.Markdown("- **manual_rect**: 手动指定矩形区域")

            with gr.Group():
                inpaint_method = gr.Radio(
                    choices=["telea", "ns"],
                    value="telea",
                    label="填充算法",
                )
                radius = gr.Slider(1, 15, value=3, step=1, label="填充半径（越大越平滑）")

            with gr.Group(visible=True) as auto_group:
                contour_area = gr.Slider(10, 500, value=50, step=10, label="最小文字面积")
                block_size = gr.Slider(3, 51, value=31, step=2, label="检测块大小")

            with gr.Group(visible=False) as manual_group:
                gr.Markdown("### 输入水印矩形区域坐标")
                gr.Markdown("用截图工具查看坐标，格式：(左上x, 左上y, 右下x, 右下y)")
                with gr.Row():
                    x1 = gr.Number(label="x1 (左上)", value=0, precision=0)
                    y1 = gr.Number(label="y1 (左上)", value=0, precision=0)
                with gr.Row():
                    x2 = gr.Number(label="x2 (右下)", value=500, precision=0)
                    y2 = gr.Number(label="y2 (右下)", value=100, precision=0)

            btn = gr.Button("🚀 开始去水印", variant="primary", size="lg")

        with gr.Column():
            output_img = gr.Image(label="去水印结果", type="numpy", height=400)
            download_btn = gr.DownloadButton(
                label="💾 下载结果",
                value=None,
            )

    # Toggle visibility based on mode
    def toggle_mode(m):
        if m == "auto_text":
            return {auto_group: gr.Group(visible=True), manual_group: gr.Group(visible=False)}
        else:
            return {auto_group: gr.Group(visible=False), manual_group: gr.Group(visible=True)}

    mode.change(toggle_mode, inputs=mode, outputs=[auto_group, manual_group])

    def process_and_save(image, mode, method, radius, area, block, cx1, cy1, cx2, cy2):
        result = process_image(image, mode, method, radius, area, block, cx1, cy1, cx2, cy2)
        if result is not None:
            out_path = Path("output") / "result.png"
            out_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            return result, str(out_path)
        return None, None

    btn.click(
        process_and_save,
        inputs=[input_img, mode, inpaint_method, radius, contour_area, block_size, x1, y1, x2, y2],
        outputs=[output_img, download_btn],
    )

    gr.Markdown("---")
    gr.Markdown("💡 提示：手动指定区域的效果通常比自动检测好很多。坐标可以用截图工具量出来。")
    gr.Markdown("GitHub: https://github.com/phoenix-zcs/dewet")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
