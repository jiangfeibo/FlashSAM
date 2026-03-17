import os
import json
import cv2
import numpy as np
from glob import glob
from pycocotools import mask as maskUtils
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm   # 进度条

def sa1b_to_yolo(json_path, output_dir, class_id=0):
    """
    将 SA-1B 标注文件转换为 YOLO Segmentation 格式
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    img_w = data["image"]["width"]
    img_h = data["image"]["height"]
    file_name = os.path.splitext(data["image"]["file_name"])[0]
    out_path = os.path.join(output_dir, f"{file_name}.txt")

    yolo_labels = []

    for ann in data["annotations"]:
        rle = ann["segmentation"]
        mask = maskUtils.decode(rle)  # (H, W) 二值掩码

        # 提取轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 3:  # 至少需要3个点
                continue

            contour = contour.squeeze(1)  # (N, 2)
            norm_points = []
            for x, y in contour:
                norm_points.append(x / img_w)
                norm_points.append(y / img_h)

            # YOLO: class_id + polygon
            line = str(class_id) + " " + " ".join(map(lambda v: f"{v:.6f}", norm_points))
            yolo_labels.append(line)

    # 保存标签
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(yolo_labels))

    return json_path  # 返回文件名，方便进度条更新


def batch_convert(input_dir, output_dir, class_id=0, num_workers=8):
    json_files = glob(os.path.join(input_dir, "*.json"))
    print(f"发现 {len(json_files)} 个 JSON 文件，开始转换...")

    os.makedirs(output_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(sa1b_to_yolo, jf, output_dir, class_id) for jf in json_files]

        # tqdm 进度条
        for _ in tqdm(as_completed(futures), total=len(futures), desc="转换进度", ncols=80):
            pass


if __name__ == "__main__":
    input_dir = "sa_1b/json_labels"   # 存放 SA-1B json 的目录
    output_dir = "sa_1b/labels"       # 生成的 YOLO 标签目录
    batch_convert(input_dir, output_dir, class_id=0)
