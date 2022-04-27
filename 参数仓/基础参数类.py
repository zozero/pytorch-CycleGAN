import argparse
import os.path

import torch.cuda

import 模型仓
from 工具仓 import 工具函数


class 基础参数:
    def __init__(self):
        self.是否已初始化 = False

    def 进行初始化(self, 解析器):
        # 基础参数
        解析器.add_argument('--数据根目录', required=True, help='图片库路径（应该包含子文件夹 trainA, trainB, valA, valB, 等等）')
        解析器.add_argument('--名称', type=str, default='实验项目', help='目录名，它决定在哪里存储类型的结果样本和给网络模型')  # 有待进一步确认意义
        解析器.add_argument('--使用权重和偏差数据库', action='store_true', help='使用权重和偏差数据库')
        解析器.add_argument('--图形处理单元标识码', type=str, default='0', help='图形处理单元标识码：例如 0，0,1,2，0,2。如果是-1则使用中央处理单元')
        解析器.add_argument('--检查点目录', type=str, default='./检查点仓', help='网络模型保存在这里')
        # 模型参数
        解析器.add_argument('--模型', type=str, default='循环生成式对抗神经网络',
                         help='选择使用哪一个神经网络模型。[循环生成式对抗神经网络|pix2pix | test | colorization]')  # 这里只会改编循环生成性对抗神经网络
        解析器.add_argument('--输入的通道数', type=int, default=3, help='输入图像的通道数：3表示rgb，1表示灰度图')
        解析器.add_argument('--输出的通道数', type=int, default=3, help='输出图像的通道数：3表示rgb，1表示灰度图')
        解析器.add_argument('--生成性对抗神经网络过滤器长度', type=int, default='64', help='在最后卷积层的生成性对抗神经网络过滤器')  # 有待进一步确认意义
        解析器.add_argument('--判别过滤器长度', type=int, default=64, help='在开头卷积层的判别过滤器')  # 有待进一步确认意义
        解析器.add_argument('--判别器的模型结构', type=str, default='基础结构',
                         help='指定判别器的模型结构 [基础结构 | n_layers | pixel]。这里基础模型结构是一个70×70补丁版生成性对抗神经网络')
        解析器.add_argument('--生成器的模型结构', type=str, default='9块版残差神经网络',
                         help='指定生成器的模型结构 [9块版残差神经网络 | 6块版残差神经网络  | unet_256 | unet_128]')
        解析器.add_argument('--n_layers_D', type=int, default=3, help='只有在判别器的模型结构等于n_layers值时使用')
        解析器.add_argument('--归一化模式', type=str, default='实例', help='实例归一化或者批归一化 [实例 | 批 | none]')
        解析器.add_argument('--网络初始化类型', type=str, default='常规类型', help='网络初始化类型[常规类型 | xavier | kaiming | orthogonal]')
        解析器.add_argument('--网络初始化比例因子', type=float, default=0.02, help='常规模式,xavier和orthogonal的比例因子')  # 有待进一步确认意义
        解析器.add_argument('--无失活率', action='store_true', help='生成器无失活率')
        # 数据集参数
        解析器.add_argument('--数据集模式', type=str, default='凌乱', help='选择数据集载入方式。 [凌乱 | aligned | single | colorization]')
        解析器.add_argument('--方向', type=str, default='A到B', help='A到B或者B到A')
        解析器.add_argument('--是否按顺序一批拿取', action='store_true', help='如果为真图片按批拿取，否则随机拿取它们')
        解析器.add_argument('--线程数', type=int, default=4, help='加载数据的线程数量')
        解析器.add_argument('--每批数量', type=int, default=1, help='每批数量')
        解析器.add_argument('--载入后尺寸', type=int, default=286, help='图像将缩放到该尺寸')
        解析器.add_argument('--裁剪后尺寸', type=int, default=256, help='然后裁剪到这个尺寸')
        解析器.add_argument('--数据集最大长度', type=int, default=float('inf'),
                         help='每个数据集允许的最大样本数。如果数据集目录包含超过 数据集最大长度，则仅加载一个子集。')
        解析器.add_argument('--图像预处理', type=str, default='重置和裁剪',
                         help='在加载时缩放和裁剪图像方法[重置和裁剪 | crop | scale_width | scale_width_and_crop | none]')
        解析器.add_argument('--不翻转', action='store_true', help='如果指定，那么不要为图像数据增强做翻转')
        解析器.add_argument('--显示时窗口的大小', type=int, default=256, help='可视化控制类和网页的显示窗口大小')
        # 额外参数
        解析器.add_argument('--当前迭代数', type=str, default='latest', help='加载到哪个迭代数？设置为最新以使用最新的缓存模型')
        解析器.add_argument('--载入迭代', type=int, default=0,
                         help='加载哪次迭代的权重网络模型？如果 载入迭代 > 0，代码将通过 iter_[load_iter] 加载模型；否则，代码将按 [当前迭代数]加载模型')
        解析器.add_argument('--冗余信息', action='store_true', help='如果指定，打印更多调试信息')
        解析器.add_argument('--后缀', type=str, default='', help='定制后缀：解析器.名称=解析器.名称+后缀 之类的{模型}_{生成器的模型结构}_尺寸{载入后尺寸}')

        self.是否已初始化 = True

        return 解析器

    def 采集参数(self):
        if not self.是否已初始化:
            解析器 = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            解析器 = self.进行初始化(解析器)

        参数, _ = 解析器.parse_known_args()
        模型名 = 参数.模型
        模型选项设置器 = 模型仓.获得选项设置器(模型名)
        解析器 = 模型选项设置器(解析器, self.是否为训练模式)

        self.解析器 = 解析器
        return 解析器.parse_args()

    def 打印参数(self, 参数):
        消息 = ''
        消息 += '----------------- 参数 ---------------\n'
        for k, v in sorted(vars(参数).items()):
            注释 = ''
            默认 = self.解析器.get_default(k)
            if v != 默认:
                注释 = '\t[默认值：%s]' % str(默认)
            消息 += '{:>25}: {:<30}{}\n'.format(str(k), str(v), 注释)
        消息 += '----------------- 结束 -------------------'
        print(消息)

        实验项目目录 = os.path.join(参数.检查点目录, 参数.名称)
        工具函数.新建多个文件夹(实验项目目录)
        文件名 = os.path.join(实验项目目录, '{}_参数.txt'.format(参数.阶段))
        with open(文件名, 'wt') as 参数文件:
            参数文件.write(消息)
            参数文件.write('\n')

    def 解析(self):
        参数 = self.采集参数()
        参数.是否为训练模式 = self.是否为训练模式

        if 参数.后缀:
            后缀 = ('' + 参数.后缀.format(**vars(参数))) if 参数.后缀 != '' else ''
            参数.名称 = 参数.名称 + 后缀

        self.打印参数(参数)

        标识列表 = 参数.图形处理单元标识码.split(',')
        参数.图形处理单元标识码 = []
        for 标识 in 标识列表:
            标识 = int(标识)
            if 标识 >= 0:
                参数.图形处理单元标识码.append(标识)
        if len(参数.图形处理单元标识码) > 0:
            torch.cuda.set_device(参数.图形处理单元标识码[0])

        self.参数 = 参数
        return self.参数
