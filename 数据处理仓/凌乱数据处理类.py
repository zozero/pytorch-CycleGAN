import os.path
import random
from PIL import Image

from 数据处理仓.图片文件夹处理类 import 制作数据集
from 数据处理仓.基础数据处理类 import 基础数据处理, 进行转化


class 凌乱数据处理(基础数据处理):
    def __init__(self, 选项):
        super().__init__(选项)
        self.目录A = os.path.join(选项.数据根目录, 选项.阶段 + 'A')
        self.目录B = os.path.join(选项.数据根目录, 选项.阶段 + 'B')

        self.A的路径列表 = sorted(制作数据集(self.目录A, 选项.数据集最大长度))
        self.B的路径列表 = sorted(制作数据集(self.目录B, 选项.数据集最大长度))
        self.A的长度 = len(self.A的路径列表)
        self.B的长度 = len(self.B的路径列表)
        是否为B到A = self.选项.方向 == 'B到A'
        输入的通道数 = self.选项.输出的通道数 if 是否为B到A else self.选项.输入的通道数
        输出的通道数 = self.选项.输入的通道数 if 是否为B到A else self.选项.输出的通道数
        self.转化A = 进行转化(self.选项, 是否转换为灰度图=(输入的通道数 == 1))
        self.转化B = 进行转化(self.选项, 是否转换为灰度图=(输出的通道数 == 1))

    def __getitem__(self, 索引):
        A路径 = self.A的路径列表[索引 % self.A的长度]
        if self.选项.是否按批拿取:
            索引B = 索引 % self.B的长度
        else:
            索引B = random.randint(0, self.B的长度 - 1)
        B路径 = self.B的路径列表[索引B]
        A图片 = Image.open(A路径).convert('RGB')
        B图片 = Image.open(B路径).convert('RGB')
        A = self.转化A(A图片)
        B = self.转化B(B图片)

        return {'A': A, 'B': B, 'A路径': A路径, 'B路径': B路径}

    def __len__(self):
        return max(self.A的长度, self.B的长度)
