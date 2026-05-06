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


def process_brush(image, mask, method, radius):
    if image is None or mask is None:
        return None, None
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract mask from editor
    if isinstance(mask, dict):
        bg = mask.get("background")
        layers = mask.get("layers", [])
        if bg is not None and len(layers) > 0:
            overlay = layers[0]
            diff = cv2.absdiff(bg, overlay)
            gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            _, mask_arr = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        else:
            mask_arr = np.zeros(img.shape[:2], dtype=np.uint8)
    elif isinstance(mask, np.ndarray):
        if mask.shape[-1] == 4:
            mask_arr = mask[:, :, 3]
        else:
            gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            _, mask_arr = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    else:
        return image, None

    if np.sum(mask_arr) == 0:
        return image, None

    result = remove_watermark(img, mask_arr, method=method, radius=radius)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), None


def process_auto(image, method, radius, contour_area, block_size):
    if image is None:
        return None, None
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = remove_text_watermark(
        img, method=method, radius=radius,
        min_contour_area=contour_area, adaptive_blocksize=block_size,
    )
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), None


def save_result(result):
    if result is None:
        return None
    out = Path("output") / "result.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return str(out)


with gr.Blocks(title="dewet - 图片去水印") as demo:

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
            input_img = gr.Image(label="上传图片", type="numpy", height=450)

            mask_editor = gr.ImageEditor(label="🎨 涂抹水印区域（用红色画笔涂）", height=450)

            with gr.Group(visible=True) as brush_group:
                gr.Markdown("💡 操作：先上传图片 → 在画布上用红色画笔涂抹水印 → 点按钮")
                gr.Markdown("💡 涂抹范围可以比水印大一点，效果会更好")

            with gr.Group(visible=False) as auto_group:
                contour_area = gr.Slider(10, 500, value=50, step=10, label="最小文字面积")
                block_size = gr.Slider(3, 51, value=31, step=2, label="检测块大小")

            with gr.Group():
                inpaint_method = gr.Radio(choices=["telea", "ns"], value="telea", label="填充算法")
                radius = gr.Slider(1, 15, value=5, step=1, label="填充半径（越大越平滑）")

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

    def run(image, mask, m, method, rad, area, block):
        if m == "brush":
            result, _ = process_brush(image, mask, method, rad)
        else:
            result, _ = process_auto(image, method, rad, area, block)
        out = save_result(result)
        return result, out

    btn.click(
        run,
        inputs=[input_img, mask_editor, mode, inpaint_method, radius, contour_area, block_size],
        outputs=[output_img, download_btn],
    )

    gr.Markdown("---")
    gr.Markdown("GitHub: https://github.com/phoenix-zcs/dewet")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
