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

        self.损失值名称列表 = []
        self.模型名称列表 = []
        self.可视化图片名列表 = []
        self.优化器列表 = []
        self.图片路径列表 = []
        self.公制 = 0  # 用于学习率策略“高原” 这是什么有待进一步分析

    @staticmethod
    def 修改命令行选项(选项, 是否为训练模式):
        return 选项

    @abstractmethod
    def 设置输入(self, 输入):
        # 从数据加载器中解压输入数据并执行必要的预处理步骤
        pass

    @abstractmethod
    def 计算前向传播(self):
        pass

    @abstractmethod
    def 计算优化器参数(self):
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

    def 评估(self):
        for 名称 in self.模型名称列表:
            if isinstance(名称, str):
                网络 = getattr(self, 名称)
                网络.eval()

    def 测试(self):
        with torch.no_grad():
            self.计算前向传播()
            self.计算视觉效果()

    def 计算视觉效果(self):
        pass

    def 取得图片路径(self):
        return self.图片路径列表

    def 更新学习率(self):
        老学习率 = self.优化器列表[0].param_groups[0]['lr']
        for 调度器 in self.调度器列表:
            if self.选项.学习率策略 == 'plateau':
                调度器.step(self.公制)
            else:
                调度器.step()
        学习率 = self.优化器列表[0].param_groups[0]['lr']
        print('学习率 %.7f -> %.7f' % (老学习率, 学习率))

    def 获得当前视觉效果(self):
        视觉效果返回值 = OrderedDict()
        for 名称 in self.可视化图片名列表:
            if isinstance(名称, str):
                视觉效果返回值[名称] = getattr(self, 名称)
        return 视觉效果返回值

    def 获得当前损失值(self):
        损失值字典返回值 = OrderedDict()
        for 名称 in self.损失值名称列表:
            if isinstance(名称, str):
                损失值字典返回值[名称] = float(getattr(self, 名称 + '的损失值'))
        return 损失值字典返回值

    def 保存神经网络(self, 后缀):
        for 名称 in self.模型名称列表:
            if isinstance(名称, str):
                保存文件名 = '%s_%s.pth' % (名称, 后缀)
                保存路径 = os.path.join(self.保存目录, 保存文件名)
                网络 = getattr(self, 名称)

                if len(self.图形处理单元标识码) > 0 and torch.cuda.is_available():
                    torch.save(网络.module.cpu().state_dict(), 保存路径)
                    网络.cuda(self.图形处理单元标识码[0])
                else:
                    torch.save(网络.cpu().state_dict(), 保存路径)

    def __修补规范的实例以兼容状态字典(self, 状态字典, 模型, 键值列表, 索引=0):
        键值 = 键值列表[索引]
        if 索引 + 1 == len(键值列表):
            if 模型.__class__.__name__.startswith('InstanceNorm') and (键值 == 'running_mean' or 键值 == 'running_var'):
                if getattr(模型, 键值) is None:
                    状态字典.pop('.'.join(键值列表))
            if 模型.__class__.__name__.startswith('InstanceNorm') and (键值 == 'num_batches_tracked'):
                状态字典.pop('.'.join(键值列表))
        else:
            self.__修补规范的实例以兼容状态字典(状态字典, getattr(模型, 键值), 键值列表, 索引 + 1)

    def 载入神经网络(self, 轮回的位子):
        # 从磁盘加载所有网络
        for 名称 in self.模型名称列表:
            if isinstance(名称, str):
                生成的文件名 = '%s_%s.pth' % (名称, 轮回的位子)
                生成的路径 = os.path.join(self.保存目录, 生成的文件名)
                网络 = getattr(self, 名称)
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
                网络 = getattr(self, 模型名)
                参数数量 = 0
                for 参数 in 网络.parameters():
                    参数数量 += 参数.numel()
                if 冗余信息:
                    print(网络)
                print('[%s] 参数全部数量 : %.3f M' % (模型名, 参数数量 / 1e6))  # 有待运行后进一步查看
        print('-----------------------------------------------')

    def 设置需要的梯度(self, 网络列表, 是否需要梯度=False):
        if not isinstance(网络列表, list):
            网络列表 = [网络列表]
        for 网络 in 网络列表:
            if 网络 is not None:
                for 参数 in 网络.parameters():
                    参数.requires_grad = 是否需要梯度
