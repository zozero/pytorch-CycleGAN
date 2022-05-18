import torch.nn

import itertools
from 工具仓.图像池塘类 import 图像池塘
from 模型仓 import 多个神经网络
from 模型仓.基础模型类 import 基础模型


class 循环生成式对抗神经网络模型(基础模型):
    @staticmethod
    def 修改命令行选项(解析器, 是否为训练模式=True):
        解析器.set_defaults(无失活率=True)
        if 是否为训练模式:
            解析器.add_argument('--lambda_A', type=float, default=10.0, help='循环损失权重 (A -> B -> A)')
            解析器.add_argument('--lambda_B', type=float, default=10.0, help='循环损失权重 (B -> A -> B)')
            解析器.add_argument('--lambda_自身', type=float, default=0.5,
                             help='使用自身映射。 将 lambda_自身 设置为 0 以外的值具有缩放自身映射损失权重的效果。 '
                                  '例如，如果自身损失的权重应该比重建损失的权重小 10 倍，请设置 lambda_自身 = 0.1，'
                                  '自身就是把生成出的图像放到生成器中生成一个新图像，计算两图像的损失值，损失值越小越好')
        return 解析器

    def __init__(self, 选项):
        基础模型.__init__(self, 选项)
        # 指定要打印的训练产生的损失值。“自身”是指 生成图B->生成器A到B->生成图B
        self.损失值名称列表 = ['判别器A', '生成器A', '循环网络A', '自身A', '判别器B', '生成器B', '循环网络B', '自身B']
        # 指定要保存/显示的图片。（重建 reconstruction images  重新构建的图片）
        可视化图片名列表A = ['原始图A', '生成图B', '还原图A']
        可视化图片名列表B = ['原始图B', '生成图A', '还原图B']
        if self.是否为训练模式 and self.选项.lambda_自身 > 0.0:
            可视化图片名列表A.append('自身B')
            可视化图片名列表B.append('自身A')

        self.可视化图片名列表 = 可视化图片名列表A + 可视化图片名列表B
        if self.是否为训练模式:
            self.模型名称列表 = ['生成器网络A', '生成器网络B', '判别器网络A', '判别器网络B']
        else:  # 测试时只需生成器
            self.模型名称列表 = ['生成器网络A', '生成器网络B']
        # 定义网络
        self.生成器网络A = 多个神经网络.定义生成器(选项.输入的通道数, 选项.输出的通道数, 选项.生成器末尾过滤器数量, 选项.生成器模型类型, 选项.归一化类型, not 选项.无失活率, 选项.网络初始化类型,
                                   选项.初始化比例因子, self.图形处理单元标识码)
        self.生成器网络B = 多个神经网络.定义生成器(选项.输入的通道数, 选项.输出的通道数, 选项.生成器末尾过滤器数量, 选项.生成器模型类型, 选项.归一化类型, not 选项.无失活率, 选项.网络初始化类型,
                                   选项.初始化比例因子, self.图形处理单元标识码)

        if self.是否为训练模式:
            self.判别器网络A = 多个神经网络.定义判别器(选项.输入的通道数, 选项.生成器末尾过滤器数量, 选项.判别器模型类型, 选项.判别器卷积层数量, 选项.归一化类型, 选项.网络初始化类型,
                                       选项.初始化比例因子, self.图形处理单元标识码)
            self.判别器网络B = 多个神经网络.定义判别器(选项.输入的通道数, 选项.生成器末尾过滤器数量, 选项.判别器模型类型, 选项.判别器卷积层数量, 选项.归一化类型, 选项.网络初始化类型,
                                       选项.初始化比例因子, self.图形处理单元标识码)

        if self.是否为训练模式:
            if 选项.lambda_自身 > 0:
                assert (选项.输入的通道数 == 选项.输出的通道数)
            self.生成图A的池塘 = 图像池塘(选项.池塘大小)  # 创建图像缓冲区以存储先前生成的图像
            self.生成图B的池塘 = 图像池塘(选项.池塘大小)
            # 定义损失值函数
            self.标准生成式对抗神经网络损失值函数 = 多个神经网络.生成式对抗神经网络损失值函数(选项.生成式对抗神经网络损失值类型).to(self.设备)
            self.标准循环网络损失值函数 = torch.nn.L1Loss()
            self.标准自身损失值函数 = torch.nn.L1Loss()
            # 初始化优化器，调度器将由函数<基础模型.setup>自动实现
            self.生成器的优化器 = torch.optim.Adam(itertools.chain(self.生成器网络A.parameters(), self.生成器网络B.parameters()),
                                            lr=选项.学习率, betas=(选项.贝塔值1, 0.999))
            self.判别器的优化器 = torch.optim.Adam(itertools.chain(self.判别器网络A.parameters(), self.判别器网络B.parameters()),
                                            lr=选项.学习率, betas=(选项.贝塔值1, 0.999))
            self.优化器列表.append(self.生成器的优化器)
            self.优化器列表.append(self.判别器的优化器)

    def 设置输入(self, 输入):
        """
        从数据加载器中解压输入数据并执行必要的预处理步骤
        :param 输入:
        :return:
        """
        A到B = self.选项.方向 == 'A到B'
        self.原始图A = 输入['A' if A到B else 'B'].to(self.设备)
        self.原始图B = 输入['B' if A到B else 'A'].to(self.设备)
        self.图片路径列表 = 输入['A路径' if A到B else 'B路径']

    def 计算前向传播(self):
        self.生成图B = self.生成器网络A(self.原始图A)
        self.还原图A = self.生成器网络B(self.生成图B)
        self.生成图A = self.生成器网络B(self.原始图B)
        self.还原图B = self.生成器网络A(self.生成图A)

    def 计算生成器的后向传播(self):
        lambda_自身 = self.选项.lambda_自身
        lambda_A = self.选项.lambda_A
        lambda_B = self.选项.lambda_B

        if lambda_自身 > 0:
            self.自身A = self.生成器网络A(self.原始图B)
            self.自身A的损失值 = self.标准自身损失值函数(self.自身A, self.原始图B)
            self.自身B = self.生成器网络B(self.原始图A)
            self.自身B的损失值 = self.标准自身损失值函数(self.自身B, self.原始图A)
        else:
            self.自身A的损失值 = 0
            self.自身B的损失值 = 0

        # 这里的目的是从生成器角度出发，它需要骗过判别器，所以判别器的结果到真值的距离就成为损失值的计算输入
        self.生成器A的损失值 = self.标准生成式对抗神经网络损失值函数(self.判别器网络A(self.生成图B), True)
        self.生成器B的损失值 = self.标准生成式对抗神经网络损失值函数(self.判别器网络B(self.生成图A), True)

        self.循环网络A的损失值 = self.标准循环网络损失值函数(self.还原图A, self.原始图A) * lambda_A
        self.循环网络B的损失值 = self.标准循环网络损失值函数(self.还原图B, self.原始图B) * lambda_B

        self.生成器的损失值 = self.生成器A的损失值 + self.生成器B的损失值 + self.循环网络A的损失值 + self.循环网络B的损失值 + self.自身A的损失值 + self.生成器B的损失值
        self.生成器的损失值.backward()

    def 计算判别器基本的后向传播(self, 判别器网络, 原始图, 生成图):
        原始图的预测图 = 判别器网络(原始图)
        原始图的判别器损失值 = self.标准生成式对抗神经网络损失值函数(原始图的预测图, True)
        生成图的预测图 = 判别器网络(生成图.detach())
        生成图的判别器损失值 = self.标准生成式对抗神经网络损失值函数(生成图的预测图, False)

        # 计算损失和计算梯度
        判别器损失值 = (原始图的判别器损失值 + 生成图的判别器损失值) * 0.5
        判别器损失值.backward()
        return 判别器损失值

    def 计算判别器A的后向传播(self):
        生成图B = self.生成图B的池塘.查询(self.生成图B)
        self.判别器A的损失值 = self.计算判别器基本的后向传播(self.判别器网络A, self.原始图B, 生成图B)

    def 计算判别器B的后向传播(self):
        生成图A = self.生成图A的池塘.查询(self.生成图A)
        self.判别器B的损失值 = self.计算判别器基本的后向传播(self.判别器网络B, self.原始图A, 生成图A)

    def 计算优化器参数(self):
        self.计算前向传播()
        self.设置需要的梯度([self.判别器网络A, self.判别器网络B], False)  # 优化 生成器 时 判别器 不需要梯度
        self.生成器的优化器.zero_grad()
        self.计算生成器的后向传播()
        self.生成器的优化器.step()

        self.设置需要的梯度([self.判别器网络A, self.判别器网络B], True)
        self.判别器的优化器.zero_grad()
        self.计算判别器A的后向传播()
        self.计算判别器B的后向传播()
        self.判别器的优化器.step()
