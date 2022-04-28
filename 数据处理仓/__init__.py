import importlib

import torch.utils.data

from 数据处理仓.基础数据处理类 import 基础数据处理


def 找到数据处理类使用名(处理模式名):
    数据处理类文件名 = "数据处理仓." + 处理模式名 + "数据处理类"
    数据集库 = importlib.import_module(数据处理类文件名)

    数据处理类 = None
    目标数据处理类名 = 处理模式名.replace('_', '') + '数据处理'
    for 类名, 类 in 数据集库.__dict__.items():
        if 类名 == 目标数据处理类名 and issubclass(类, 基础数据处理):
            数据处理类 = 类

    if 数据处理类 is None:
        raise NotImplementedError("错误：找到数据处理类使用名，数据处理类文件名：%s.py，目标数据处理类名：%s" % (数据处理类文件名, 目标数据处理类名))

    return 数据处理类


def 创建数据集(选项):
    数据载入器 = 用户数据载入器(选项)
    数据集 = 数据载入器.载入数据()
    return 数据集


class 用户数据载入器:
    def __init__(self, 选项):
        self.选项 = 选项
        数据处理类 = 找到数据处理类使用名(self.选项.数据处理模式)
        self.数据集 = 数据处理类(选项)
        print('数据集[%s]已被创建' % type(self.数据集).__name__)
        self.数据载入器 = torch.utils.data.DataLoader(
            self.数据集,
            batch_size=选项.每批数量,
            shuffle=not 选项.是否按批拿取,
            num_workers=int(选项.线程数)
        )

    def 载入数据(self):
        return self

    def __len__(self):
        return min(len(self.数据集), self.选项.数据集最大长度)

    def __iter__(self):
        for i, 数据集 in enumerate(self.数据载入器):
            if i * self.选项.每批数量 >= self.选项.数据集最大长度:
                break
            yield 数据集
