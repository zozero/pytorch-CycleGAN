import os.path
from abc import ABC, abstractmethod
from collections import OrderedDict

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
        if not self.是否为训练模式 or 选项.是否继续训练:
            加入后缀 = '迭代_%d' % 选项.迭代的位子 if 选项.迭代的位子 > 0 else 选项.轮回的位子
            self.载入神经网络(加入后缀)
        self.打印神经网络(选项.冗余信息)

    def 载入神经网络(self, 轮回的位子):
        # 从磁盘加载所有网络
        for 名称 in self.模型名称列表:
            if isinstance(名称, str):
                生成的文件名 = '%s的第%s轮' % (名称, 轮回的位子)
                生成的路径 = os.path.join(self.保存目录, 生成的文件名)
                网络 = getattr(self, 名称 + '的网络')
                if isinstance(网络, torch.nn.DataParallel):
                    网络 = 网络.module
                print('从%s载入模型' % 生成的路径)
                状态字典 = torch.load(生成的路径, map_location=self.设备)
                if hasattr(状态字典, '_metadata'):
                    del 状态字典._metadata
                for 键值 in list(状态字典.keys()):
                    self.__修补规范的实例以兼容状态字典(状态字典, 网络, 键值.split('.'))
                网络.load_state_dict(状态字典)

    def 打印神经网络(self, 冗余信息):
        print('---------- 已初始化神经网络 -------------')
        for 模型名 in self.模型名称列表:
            if isinstance(模型名, str):
                网络 = getattr(self, 模型名 + '的网络')
                参数数量 = 0
                for 参数 in 网络.parameters():
                    参数数量 += 参数.numel()
                if 冗余信息:
                    print(网络)
                print('[神经网络 %s] 参数全部数量 : %.3f M' % (模型名, 参数数量 / 1e6))  # 有待运行后进一步查看
        print('-----------------------------------------------')

    def __修补规范的实例以兼容状态字典(self, 状态字典, 模型, 键值列表, 索引=0):
        键值 = 键值列表[索引]
        if 索引 + 1 == len(键值列表):
            if 模型.__class__.__name__.startswitch('InstanceNorm') and (键值 == 'running_mean' or 键值 == 'running_var'):
                if getattr(模型, 键值) is None:
                    状态字典.pop('.'.join(键值列表))
            if 模型.__class__.__name__.startswitch('InstanceNorm') and (键值 == 'num_batches_tracked'):
                状态字典.pop('.'.join(键值列表))
        else:
            self.__修补规范的实例以兼容状态字典(状态字典, getattr(模型, 键值), 键值列表, 索引 + 1)
