import time

from 工具仓.可视化工具类 import 可视化工具
from 模型仓 import 创建模型
from 选项仓.训练用选项类 import 训练用选项
from 数据处理仓 import 创建数据集

if __name__ == '__main__':
    训练用选项 = 训练用选项().解析()
    数据集 = 创建数据集(训练用选项)
    数据集长度 = len(数据集)
    print('训练时图像的数量=%d' % 数据集长度)

    模型 = 创建模型(训练用选项)
    模型.设置(训练用选项)
    可视化工具实例 = 可视化工具(训练用选项)
    完成的迭代数 = 0

    for 轮回索引 in range(训练用选项.轮回起始数, 训练用选项.总轮回数 + 训练用选项.轮回衰减数 + 1):
        轮回开始时间 = time.time()
        迭代数据时间 = time.time()
        单次轮回的迭代数 = 0  # 当前轮回的训练迭代次数，每个轮回重置为 0
        可视化工具实例.重置()
        # 这里存在一个报错，可能需要更改代码的位置
        模型.更新学习率()
        for i, 数据 in enumerate(数据集):
            迭代开始时间 = time.time()
            if 完成的迭代数 % 训练用选项.控制台更新频率 == 0:
                时间记录 = 迭代开始时间 - 迭代数据时间
            完成的迭代数 += 训练用选项.每批数量
            单次轮回的迭代数 += 训练用选项.每批数量
            模型.设置输入(数据)
            模型.计算优化器参数()  # 计算损失函数，获取梯度，更新网络权重
        if 单次轮回的迭代数 % 训练用选项.显示的频率 == 0:
            是否保存结果 = 单次轮回的迭代数 % 训练用选项.更新超文本标记语言频率 == 0
            模型.计算视觉效果()
            可视化工具实例.显示当前结果(模型.获得当前视觉效果(), 轮回索引, 是否保存结果)

        if 单次轮回的迭代数%训练用选项.控制台更新频率 == 0:
            pass
        exit()
