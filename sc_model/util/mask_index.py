# -*- coding = utf-8 -*-
# @Date: 2023/10/20
# @Time: 21:39
# @Author:tsw
# @File：main.py
# @Software: PyCharm
# @description: 生成noise用于mask采样，优先选择中心位于检测框内的patch。adaptRatio：控制优先编码框内patch的比例
import torch
import random


def generate_random_numbers(images, coordinates, adaptRatio):
    # random.seed(3407)

    B, C, H, W = images.size()

    random_numbers = torch.zeros(B, 196)
    for b in range(B):
        xmin, ymin, xmax, ymax = coordinates[b]
        for i in range(196):
            row, col = divmod(i, 14)
            center_x = (col + 0.5) * (W / 16)
            center_y = (row + 0.5) * (H / 16)
            random_num = random.uniform(0, 1)

            # 检查中心坐标是否在边界框内
            if xmin <= center_x <= xmax and ymin <= center_y <= ymax:
                if random_num < adaptRatio:
                    random_number = random.random()
                else:
                    random_number = random.random() + 1
            else:
                if random_num < adaptRatio:
                    random_number = random.random() + 1
                else:
                    random_number = random.random()

            random_numbers[b, i] = random_number

    return random_numbers


if __name__ == '__main__':
    B = 32
    C = 3
    H = 224
    W = 224

    images = torch.randn(B, C, H, W)  # 示例输入图像张量
    coordinates = torch.rand(B, 4) + 10  # 示例坐标张量
    # print(images.shape, coordinates.shape)

    random_numbers = generate_random_numbers(images, coordinates, 0.8)
    print(random_numbers.shape)
    print(random_numbers)
