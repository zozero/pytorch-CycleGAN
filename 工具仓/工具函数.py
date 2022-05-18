import os.path

import numpy as np
import torch
from PIL import Image


def 把张量转成图片(输入的图片, 图片类型=np.uint8):
    if not isinstance(输入的图片, np.ndarray):
        if isinstance(输入的图片, torch.Tensor):
            图片张量 = 输入的图片.data
        else:
            return 输入的图片
        图片矩阵 = 图片张量[0].cpu().float().numpy()  # 将其转换为 numpy 数组
        if 图片矩阵.shape[0] == 1:  # 灰度图到rgb 红绿蓝
            图片矩阵 = np.tile(图片矩阵, (3, 1, 1))
        图片矩阵 = (np.transpose(图片矩阵, (1, 2, 0)) + 1) / 2.0 * 255.0  # 后处理：转置和缩放
    else:
        图片矩阵 = 输入的图片
    return 图片矩阵.astype(图片类型)


def 保存图片(图片矩阵, 图片路径, 比率=1.0):
    图片 = Image.fromarray(图片矩阵)
    h, w, _ = 图片矩阵.shape

    if 比率 > 1.0:
        图片 = 图片.resize((h, int(w * 比率)), Image.BICUBIC)
    if 比率 < 1.0:
        图片 = 图片.resize((int(h / 比率), w), Image.BICUBIC)
    图片.save(图片路径)


def 新建多个文件夹(路径列表):
    if isinstance(路径列表, list) and not isinstance(路径列表, str):
        for 路径 in 路径列表:
            新建文件夹(路径)
    else:
        新建文件夹(路径列表)


def 新建文件夹(路径):
    if not os.path.exists(路径):
        os.makedirs(路径)
