from abc import ABC, abstractmethod


class 基础模型(ABC):
    def __init__(self, 参数):
        self.参数 = 参数
        self.图形处理单元标识码 = 参数.图形处理单元标识码
        self.是否为训练模式 = 参数.是否为训练模式

    @staticmethod
    def 修改命令行参数(参数, 是否为训练模式):
        return 参数
