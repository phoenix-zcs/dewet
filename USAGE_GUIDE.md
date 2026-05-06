# 🧹 dewet 使用指南（小白版）

> 从零开始，跟着做就行。

---

## 第一步：安装 Python

如果电脑还没装 Python，去这里下载：
👉 https://www.python.org/downloads/

安装时记得勾选 **"Add Python to PATH"**

---

## 第二步：下载项目

打开终端（Windows 按 Win+R 输入 cmd，Mac 按 command+空格输入 Terminal）：

```
git clone https://github.com/phoenix-zcs/dewet.git
cd dewet
```

---

## 第三步：安装依赖

```
pip install -r requirements.txt
```

---

## 第四步：准备一张带水印的图

把你的图片放进项目文件夹，比如叫 `test.jpg`

---

## 第五步：运行！

### 🅰 最简单：自动去除文字水印

在 `dewet` 文件夹里新建一个文件 `run.py`，写入：

```python
from dewet import load_image, remove_text_watermark, save_image

# 读取图片
img = load_image("test.jpg")

# 自动去除文字水印
result = remove_text_watermark(img)

# 保存结果
save_image("output.jpg", result)
print("搞定！结果保存在 output.jpg")
```

然后运行：
```
python run.py
```

打开 `output.jpg` 看效果 ✅

---

### 🅱 手动指定水印位置（效果更好）

如果你知道水印在图片的哪个位置，可以手动框选：

```python
from dewet import load_image, remove_watermark, create_mask, save_image

img = load_image("test.jpg")

# 用截图工具量一下水印的位置
# 格式：(x起点, y起点, 宽度, 高度)
# 比如：水印在图片右下角，距左边400像素，距上面300像素，宽200，高50
mask = create_mask(img, [(400, 300, 200, 50)])

# 去除水印
result = remove_watermark(img, mask)

# 保存
save_image("output.jpg", result)
print("搞定！")
```

**怎么知道水印的坐标？**

1. 用系统截图工具截图，看像素尺寸
2. 或者用画图软件打开图片，看水印的位置和大小
3. 不用很精确，框大一点也行

---

### 🅲 运行官方 demo（三合一）

```
python examples/demo.py test.jpg
```

会自动生成三个文件：
- `output/manual_result.jpg` — 手动遮罩结果
- `output/auto_text_result.jpg` — 自动文字检测结果
- `output/comparison.jpg` — 前后对比图

---

## 常见问题

**Q: pip install 报错？**
→ 试试 `pip3 install -r requirements.txt`

**Q: 找不到 dewet 模块？**
→ 确保你在 `dewet` 项目文件夹里运行命令

**Q: 效果不好？**
→ 手动指定位置（方式B）比自动检测效果好很多
→ 调大 `radius` 参数可以让填充更平滑：`remove_watermark(img, mask, radius=7)`

---

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| method | `telea`（快）或 `ns`（精细） | telea |
| radius | 填充半径，越大越平滑 | 3 |
| min_contour_area | 最小文字面积，越小越敏感 | 50 |

---

有问题随时问我 🐉
