import os.path
from abc import ABC, abstractmethod

import torch

from 模型仓 import 多个神经网络


class 基础模型(ABC):
    def __init__(self, 选项):
        self.选项 = 选项
        self.图形处理单元标识码 = 选项.图形处理单元标识码
        self.是否为训练模式 = 选项.是否为训练模式
        self.设备 = torch.device('cuda:{}'.format(self.图形处理单元标识码[0])) if self.图形处理单元标识码 else torch.device('cpu')
        self.保存目录 = os.path.join(选项.检查点目录, 选项.名称)
        if 选项.图像预处理 != '宽度与比例':
            torch.backends.cudnn.benchmark = True

        self.损失函数名称列表 = []
        self.模型名称列表 = []
        self.可视化名称列表 = []
        self.优化器列表 = []
        self.图片路径列表 = []
        self.公制 = 0  # 用于学习率策略“高原” 这是什么有待进一步分析

    @staticmethod
    def 修改命令行选项(选项, 是否为训练模式):
        return 选项

    @abstractmethod
    def 设置输入步骤(self):
        # 从数据加载器中解压输入数据并执行必要的预处理步骤
        pass

    @abstractmethod
    def 前向传播(self):
        pass

    @abstractmethod
    def 设置优化器参数(self):
        # 计算损失、梯度，并更新网络权重；在每次训练迭代中调用
        pass

    def 设置(self, 选项):
        # 加载和打印网络；创建调度程序
        if 选项.是否为训练模式:
            self.调度器列表 = [多个神经网络.获取调度器(优化器, 选项) for 优化器 in self.优化器列表]
