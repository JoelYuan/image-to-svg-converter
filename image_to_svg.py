import numpy as np
import cv2
import svgwrite
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def process_image(file_path, width, sample_points):
    # 生成掩码矩阵（蜘蛛网取样，自适应sigma）
    x, y = np.meshgrid(np.arange(width), np.arange(width))
    dist = np.sqrt((x - width/2)**2 + (y - width/2)**2)
    sigma = width / 4  # 自适应sigma
    prob = np.exp(-dist**2 / (2 * sigma**2))
    prob /= prob.sum()
    points = np.random.choice(width*width, size=sample_points, p=prob.flatten())
    sample_x, sample_y = np.unravel_index(points, (width, width))
    mask = np.zeros((width, width), dtype=np.uint8)
    mask[sample_y, sample_x] = 1

    # 加载并缩放图像（支持多种格式）
    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 支持中文路径
    if img is None:
        raise ValueError(f"无法加载图片: {file_path}")
    img_resized = cv2.resize(img, (width, width)).astype(np.float32)

    # 提取取样点颜色并聚类
    sample_colors = img_resized[mask == 1]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(sample_colors, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 按色相排序颜色
    def sort_by_hue(centers):
        hsv = cv2.cvtColor(centers[np.newaxis].astype(np.uint8), cv2.COLOR_BGR2HSV)[0]
        return centers[np.argsort(hsv[:, 0])]
    centers = sort_by_hue(centers)

    # 矢量化量化全图
    distances = np.linalg.norm(img_resized[:, :, np.newaxis] - centers, axis=3)
    labels_full = np.argmin(distances, axis=2)
    quantized_img = centers[labels_full].astype(np.uint8)

    # 分层提取并过滤轮廓（平滑处理）
    color_layers = [(quantized_img == color).all(axis=2).astype(np.uint8) * 255 for color in centers.astype(np.uint8)]
    filtered_contours_list = []
    epsilon = 1.0  # 平滑参数
    for mask in color_layers:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cv2.approxPolyDP(c, epsilon, True) for c in contours if cv2.contourArea(c) >= 30]
        filtered_contours_list.append(filtered_contours)

    # 生成SVG（分组增强）
    output_path = os.path.splitext(file_path)[0] + ".svg"
    dwg = svgwrite.Drawing(output_path, size=(width, width))
    for i, contours in enumerate(filtered_contours_list):
        color_rgb = centers[i].astype(int)
        fill_color = f"rgb({color_rgb[2]},{color_rgb[1]},{color_rgb[0]})"
        grp = dwg.g(id=f"color_{i}", fill=fill_color)
        for contour in contours:
            points = [(float(pt[0][0]), float(pt[0][1])) for pt in contour]
            grp.add(dwg.polygon(points))
        dwg.add(grp)
    dwg.save()

    return quantized_img, output_path

def create_gui():
    root = tk.Tk()
    root.title("图片转SVG工具")
    root.geometry("400x300")

    # 变量
    width_var = tk.StringVar(value="400")
    points_var = tk.StringVar(value="1600")
    file_path_var = tk.StringVar()

    # 文件选择
    def select_file():
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            file_path_var.set(file_path)
            file_label.config(text=f"已选择: {os.path.basename(file_path)}")

    tk.Label(root, text="选择图片:").pack(pady=5)
    tk.Button(root, text="浏览", command=select_file).pack()
    file_label = tk.Label(root, text="未选择文件")
    file_label.pack()

    # 边长输入
    tk.Label(root, text="图片边长:").pack(pady=5)
    tk.Entry(root, textvariable=width_var).pack()

    # 取样点数输入
    tk.Label(root, text="取样点数 (低于边长²/2):").pack(pady=5)
    tk.Entry(root, textvariable=points_var).pack()

    # 处理并预览
    def process_and_preview():
        file_path = file_path_var.get()
        if not file_path:
            messagebox.showerror("错误", "请先选择图片！")
            return

        try:
            width = int(width_var.get())
            points = int(points_var.get())
            max_points = (width * width) // 2
            if width <= 0 or points <= 0 or points > max_points:
                messagebox.showerror("错误", f"边长需大于0，取样点数需在1-{max_points}之间")
                return

            quantized_img, output_path = process_image(file_path, width, points)
            
            # 预览量化图像
            cv2.imshow("预览 (按任意键关闭)", quantized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            messagebox.showinfo("成功", f"SVG已保存至: {output_path}")
        except ValueError as e:
            messagebox.showerror("错误", f"输入无效: {e}")
        except Exception as e:
            messagebox.showerror("错误", f"处理失败: {e}")

    tk.Button(root, text="处理并预览", command=process_and_preview).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    create_gui()