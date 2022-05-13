import random

import torch


class 图像池塘:
    def __init__(self, 池塘大小):
        self.池塘大小 = 池塘大小
        if self.池塘大小 > 0:
            self.图像数量 = 0
            self.图像列表 = []

    def 查询(self, 复数图像):
        if self.池塘大小 == 0:
            return 复数图像
        图像返回列表 = []
        for 图像 in 复数图像:
            图像 = torch.unsqueeze(图像.data, 0)
            if self.图像数量 < self.池塘大小:
                self.图像数量 = self.图像数量 + 1
                self.图像列表.append(图像)
                图像返回列表.append(图像)
            else:
                零一 = random.uniform(0, 1)
                if 零一 > 0.5:  # 有 50% 的机会，缓冲区将返回之前存储的图像，并将当前图像插入缓冲区
                    随机标识 = random.randint(0, self.池塘大小 - 1)
                    临时 = self.图像列表[随机标识].clone()
                    self.图像列表[随机标识] = 图像
                    图像返回列表.append(临时)
                else:
                    图像返回列表.append(图像)

            图像返回列表 = torch.cat(图像返回列表, 0)
            return 图像返回列表
