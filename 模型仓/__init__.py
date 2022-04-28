import importlib

from 模型仓.基础模型类 import 基础模型


def 用模型名找到模型(模型名):
    模型文件名 = "模型仓." + 模型名 + "模型类"
    模型库 = importlib.import_module(模型文件名)
    模型 = None
    目标模型名 = 模型名 + "模型"
    for 类名, 类 in 模型库.__dict__.items():
        if 类名 == 目标模型名 and issubclass(类, 基础模型):
            模型 = 类
            break

    if 模型 is None:
        print(模型文件名 + '载入错误')
        exit(0)

    return 模型


def 获得选项设置器(模型名):
    模型类 = 用模型名找到模型(模型名)
    return 模型类.修改命令行选项


def 创建模型(选项):
    模型 = 用模型名找到模型(选项.模型)
    实例 = 模型(选项)
    print("[%s]模型已被创建" % type(实例).__name__)
    return 实例
