import functools
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.nn import init


class 自身(nn.Module):
    def forward(self, x):
        return x


def 获得层归一化(归一化类型='实例'):
    if 归一化类型 == '批':
        层归一化 = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif 归一化类型 == '实例':
        层归一化 = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif 归一化类型 == 'none':
        def 层归一化(x):
            return 自身()
    else:
        raise NotImplementedError('层归一化 [%s] 没有找到' % 归一化类型)
    return 层归一化


def 初始化权重(网络, 初始化类型='常规', 初始化比例因子=0.02):
    def 初始化函数(m):
        类名 = m.__class__.__name__
        if hasattr(m, 'weight') and (类名.find('Conv') != -1 or 类名.find('Linear') != -1):
            if 初始化类型 == '常规':
                init.normal_(m.weight.data, 0.0, 初始化比例因子)
            elif 初始化类型 == 'xavier':
                init.xavier_normal_(m.weight.data, gain=初始化比例因子)
            elif 初始化类型 == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif 初始化类型 == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=初始化比例因子)
            else:
                raise NotImplementedError('初始化方法 [%s] 没有实施' % 初始化类型)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif 类名.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 初始化比例因子)
            init.constant_(m.bias.data, 0.0)

    print('%s 初始化网络' % 初始化类型)
    网络.apply(初始化函数)


def 初始化网络(网络, 初始化类型='常规', 初始化比例因子=0.02, 图形处理单元标识码=[]):
    if len(图形处理单元标识码) > 0:
        assert (torch.cuda.is_available())
        网络.to(图形处理单元标识码[0])
        网络 = torch.nn.DataParallel(网络, 图形处理单元标识码)
    初始化权重(网络, 初始化类型=初始化类型, 初始化比例因子=初始化比例因子)

    return 网络


def 获取调度器(优化器, 选项):
    if 选项.学习率策略 == '线性':
        def lambda_规则(轮回):
            # 这个公式什么意思暂时没看懂......
            线性学习率 = 1.0 - max(0, 轮回 + 选项.轮回起始数 - 选项.轮回次数) / float(选项.轮回衰减数 + 1)
            return 线性学习率

        调度器 = lr_scheduler.LambdaLR(优化器, lr_lambda=lambda_规则)
    elif 选项.学习率策略 == 'step':
        调度器 = lr_scheduler.StepLR(优化器, step_size=选项.迭代衰减间隔, gamma=0.1)
    elif 选项.学习率策略 == 'plateau':
        调度器 = lr_scheduler.ReduceLROnPlateau(优化器, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif 选项.学习率策略 == 'cosine':
        调度器 = lr_scheduler.CosineAnnealingLR(优化器, T_max=选项.轮回次数, eta_min=0)
    else:
        return NotImplementedError("学习率策略[%s]没被实施", 选项.学习率策略)

    return 调度器


def 定义生成器(输入的通道数, 输出的通道数, 生成器过滤器数量, 生成器网络名, 归一化类型='批', 是否使用失活率=False, 网络初始化类型='常规', 初始化比例因子=0.02, 图形处理单元标识码=[]):
    网络 = None
    层归一化 = 获得层归一化(归一化类型=归一化类型)

    if 生成器网络名 == '9块版残差神经网络':
        网络 = 残差神经网络生成器(输入的通道数, 输出的通道数, 生成器过滤器数量, 层归一化=层归一化, 是否使用失活率=是否使用失活率, 块数=9)
    elif 生成器网络名 == '6块版残差神经网络':
        网络 = 残差神经网络生成器(输入的通道数, 输出的通道数, 生成器过滤器数量, 层归一化=层归一化, 是否使用失活率=是否使用失活率, 块数=6)
    elif 生成器网络名 == 'U型网络_128':
        网络 = U型网络生成器(输入的通道数, 输出的通道数, 7, 生成器过滤器数量, 层归一化=层归一化, 是否使用失活率=是否使用失活率)
    elif 生成器网络名 == 'U型网络_256':
        网络 = U型网络生成器(输入的通道数, 输出的通道数, 8, 生成器过滤器数量, 层归一化=层归一化, 是否使用失活率=是否使用失活率)
    else:
        raise NotImplementedError('生成器网络名 [%s] 不在范围中' % 生成器网络名)
    return 初始化网络(网络, 网络初始化类型, 初始化比例因子, 图形处理单元标识码)


def 定义判别器(输入的通道数, 判别器过滤器数量, 判别器网络名, 判别器卷积层数量=3, 归一化类型='批', 网络初始化类型='常规', 初始化比例因子=0.02, 图形处理单元标识码=[]):
    网络 = None
    层归一化 = 获得层归一化(归一化类型=归一化类型)

    if 判别器网络名 == '基础':
        网络 = 多层判别器(输入的通道数, 判别器过滤器数量, 层数=3, 层归一化=层归一化)
    elif 判别器网络名 == '更多层数':
        网络 = 多层判别器(输入的通道数, 判别器过滤器数量, 层数=判别器卷积层数量, 层归一化=层归一化)
    elif 判别器网络名 == '像素':  # 每个像素分类为真或假
        网络 = 像素判别器(输入的通道数, 判别器过滤器数量, 层归一化=层归一化)
    else:
        raise NotImplementedError('判别器网络名 [%s] 不在范围中' % 判别器网络名)

    return 初始化网络(网络, 初始化类型=网络初始化类型, 初始化比例因子=初始化比例因子, 图形处理单元标识码=图形处理单元标识码)


class 残差神经网络生成器(nn.Module):
    def __init__(self, 输入的通道数, 输出的通道数, 生成器过滤器数量=64, 层归一化=nn.BatchNorm2d, 是否使用失活率=False, 块数=6, 填充类型='反射'):
        assert (块数 >= 0)
        super(残差神经网络生成器, self).__init__()
        if type(层归一化) == functools.partial:
            是否使用偏差 = 层归一化.func == nn.InstanceNorm2d
        else:
            是否使用偏差 = 层归一化 == nn.InstanceNorm2d

        模型 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(输入的通道数, 生成器过滤器数量, kernel_size=7, padding=0, bias=是否使用偏差),
            层归一化(生成器过滤器数量),
            nn.ReLU(True)
        ]

        下采样数量 = 2
        for i in range(下采样数量):
            倍数 = 2 ** i
            模型 += [
                nn.Conv2d(生成器过滤器数量 * 倍数, 生成器过滤器数量 * 倍数 * 2, kernel_size=3, stride=2, padding=1, bias=是否使用偏差),
                层归一化(生成器过滤器数量 * 倍数 * 2),
                nn.ReLU(True)
            ]

        倍数 = 2 ** 下采样数量
        for i in range(块数):
            模型 += [残差神经网络块(生成器过滤器数量 * 倍数, 填充类型=填充类型, 层归一化=层归一化, 是否使用失活率=是否使用失活率, 是否使用偏差=是否使用偏差)]

        for i in range(下采样数量):
            倍数 = 2 ** (下采样数量 - i)
            模型 += [
                nn.ConvTranspose2d(
                    生成器过滤器数量 * 倍数,
                    int(生成器过滤器数量 * 倍数 / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=是否使用偏差
                ),
                层归一化(int(生成器过滤器数量 * 倍数 / 2)),
                nn.ReLU(True)
            ]
        模型 += [nn.ReflectionPad2d(3)]
        模型 += [nn.Conv2d(生成器过滤器数量, 输出的通道数, kernel_size=7, padding=0)]
        模型 += [nn.Tanh()]

        self.模型 = nn.Sequential(*模型)

    def forward(self, 输入):
        return self.模型(输入)


class 残差神经网络块(nn.Module):
    def __init__(self, 维度数, 填充类型, 层归一化, 是否使用失活率, 是否使用偏差):
        super(残差神经网络块, self).__init__()
        self.卷积块 = self.构建卷积块(维度数, 填充类型, 层归一化, 是否使用失活率, 是否使用偏差)

    def 构建卷积块(self, 维度数, 填充类型, 层归一化, 是否使用失活率, 是否使用偏差):
        卷积块 = []
        p = 0
        if 填充类型 == '反射':
            卷积块 += [nn.ReflectionPad2d(1)]
        elif 填充类型 == '复制':
            卷积块 += [nn.ReplicationPad2d(1)]
        elif 填充类型 == '零':
            p = 1
        else:
            raise NotImplementedError('填充 [%s] 没有实施' % 填充类型)
        卷积块 += [nn.Conv2d(维度数, 维度数, kernel_size=3, padding=p, bias=是否使用偏差), 层归一化(维度数), nn.ReLU(True)]
        if 是否使用失活率:
            卷积块 += [nn.Dropout(0.5)]

        p = 0
        if 填充类型 == '反射':
            卷积块 += [nn.ReflectionPad2d(1)]
        elif 填充类型 == '复制':
            卷积块 += [nn.ReplicationPad2d(1)]
        elif 填充类型 == '零':
            p = 1
        else:
            raise NotImplementedError('填充 [%s] 没有实施' % 填充类型)
        卷积块 += [nn.Conv2d(维度数, 维度数, kernel_size=3, padding=p, bias=是否使用偏差), 层归一化(维度数)]

        return nn.Sequential(*卷积块)

    def forward(self, x):
        输出 = x + self.卷积块(x)
        return 输出


class U型网络生成器(nn.Module):
    def __init__(self, 输入的通道数, 输出的通道数, 下采样数, 生成器过滤器数量=64, 层归一化=nn.BatchNorm2d, 是否使用失活率=False):
        super(U型网络生成器, self).__init__()
        U型网络块 = U型网络跳过连接块(生成器过滤器数量 * 8, 生成器过滤器数量 * 8, 输入的通道数=None, 子模块=None, 层归一化=层归一化, 是否为最内层模块=True)
        for i in range(下采样数 - 5):
            U型网络块 = U型网络跳过连接块(生成器过滤器数量 * 8, 生成器过滤器数量 * 8, 输入的通道数=None, 子模块=U型网络块, 层归一化=层归一化, 是否使用失活率=是否使用失活率)
        U型网络块 = U型网络跳过连接块(生成器过滤器数量 * 4, 生成器过滤器数量 * 8, 输入的通道数=None, 子模块=U型网络块, 层归一化=层归一化)
        U型网络块 = U型网络跳过连接块(生成器过滤器数量 * 2, 生成器过滤器数量 * 4, 输入的通道数=None, 子模块=U型网络块, 层归一化=层归一化)
        U型网络块 = U型网络跳过连接块(生成器过滤器数量, 生成器过滤器数量 * 2, 输入的通道数=None, 子模块=U型网络块, 层归一化=层归一化)
        self.模型 = U型网络跳过连接块(输出的通道数, 生成器过滤器数量, 输入的通道数=输入的通道数, 子模块=U型网络块, 是否为最外层模块=True, 层归一化=层归一化)

    def forward(self, 输入):
        return self.模型(输入)


class U型网络跳过连接块(nn.Module):
    def __init__(self, 卷积层外过滤器数量, 卷积层内过滤器数量, 输入的通道数=None, 子模块=None, 是否为最外层模块=False, 是否为最内层模块=False,
                 层归一化=nn.BatchNorm2d, 是否使用失活率=False):
        super(U型网络跳过连接块, self).__init__()
        self.是否为最外层模块 = 是否为最外层模块
        if type(层归一化) == functools.partial:
            是否使用偏差 = 层归一化.func == nn.InstanceNorm2d
        else:
            是否使用偏差 = 层归一化 == nn.InstanceNorm2d
        if 输入的通道数 is None:
            输入的通道数 = 卷积层外过滤器数量
        下卷积层 = nn.Conv2d(输入的通道数, 卷积层内过滤器数量, kernel_size=4, stride=2, padding=1, bias=是否使用偏差)
        下线性整流函数 = nn.LeakyReLU(0.2, True)
        下归一化 = 层归一化(卷积层内过滤器数量)
        上线性整流函数 = nn.ReLU(True)
        上归一化 = 层归一化(卷积层外过滤器数量)

        if 是否为最外层模块:
            上卷积层 = nn.ConvTranspose2d(卷积层内过滤器数量 * 2, 卷积层外过滤器数量, kernel_size=4, stride=2, padding=1)
            下 = [下卷积层]
            上 = [上线性整流函数, 上卷积层, nn.Tanh()]
            模型 = 下 + [子模块] + 上
        elif 是否为最内层模块:
            上卷积层 = nn.ConvTranspose2d(卷积层内过滤器数量, 卷积层外过滤器数量, kernel_size=4, stride=2, padding=1, bias=是否使用偏差)
            下 = [下线性整流函数, 下卷积层]
            上 = [上线性整流函数, 上卷积层, 上归一化]
            模型 = 下 + 上
        else:
            上卷积层 = nn.ConvTranspose2d(卷积层内过滤器数量 * 2, 卷积层外过滤器数量, kernel_size=4, stride=2, padding=1, bias=是否使用偏差)
            下 = [下线性整流函数, 下卷积层, 下归一化]
            上 = [上线性整流函数, 上卷积层, 上归一化]
            if 是否使用失活率:
                模型 = 下 + [子模块] + 上 + [nn.Dropout(0.5)]
            else:
                模型 = 下 + [子模块] + 上

        self.模型 = nn.Sequential(*模型)

    def forward(self, x):
        if self.是否为最外层模块:
            return self.模型(x)
        else:
            return torch.cat([x, self.模型(x)], 1)


class 多层判别器(nn.Module):
    def __init__(self, 输入的通道数, 过滤器数量=64, 层数=3, 层归一化=nn.BatchNorm2d):
        super(多层判别器, self).__init__()
        if type(层归一化) == functools.partial:
            是否使用偏差 = 层归一化.func == nn.InstanceNorm2d
        else:
            是否使用偏差 = 层归一化 == nn.InstanceNorm2d

        内核尺寸 = 4
        填充大小 = 1
        模型 = [nn.Conv2d(输入的通道数, 过滤器数量, kernel_size=内核尺寸, stride=2, padding=填充大小), nn.LeakyReLU(0.2, True)]
        过滤器数量乘数 = 1
        当前过滤器数量乘数 = 1
        for n in range(1, 层数):
            当前过滤器数量乘数 = 过滤器数量乘数
            过滤器数量乘数 = min(2 ** n, 8)
            模型 += [
                nn.Conv2d(过滤器数量 * 当前过滤器数量乘数, 过滤器数量 * 过滤器数量乘数, kernel_size=内核尺寸, stride=2, padding=填充大小,
                          bias=是否使用偏差),
                层归一化(过滤器数量 * 过滤器数量乘数),
                nn.LeakyReLU(0.2, True)
            ]

        当前过滤器数量乘数 = 过滤器数量乘数
        过滤器数量乘数 = min(2 ** 层数, 8)
        模型 += [
            nn.Conv2d(过滤器数量 * 当前过滤器数量乘数, 过滤器数量 * 过滤器数量乘数, kernel_size=内核尺寸, stride=1, padding=填充大小, bias=是否使用偏差),
            层归一化(过滤器数量 * 过滤器数量乘数),
            nn.LeakyReLU(0.2, True)
        ]

        模型 += [nn.Conv2d(过滤器数量 * 过滤器数量乘数, 1, kernel_size=内核尺寸, stride=1, padding=填充大小)]  # 输出单通道的预测图
        self.模型 = nn.Sequential(*模型)

    def forward(self, 输入):
        return self.模型(输入)


class 像素判别器(nn.Module):
    """定义一个1*1的补丁版生成式对抗神经网络（像素版生成式对抗神经网络）"""

    def __init__(self, 输入的通道数, 过滤器数量=64, 层归一化=nn.BatchNorm2d):
        super(像素判别器, self).__init__()
        if type(层归一化) == functools.partial:
            是否使用偏差 = 层归一化.func == nn.InstanceNorm2d
        else:
            是否使用偏差 = 层归一化 == nn.InstanceNorm2d

        self.网络 = [
            nn.Conv2d(输入的通道数, 过滤器数量, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(输入的通道数, 过滤器数量 * 2, kernel_size=1, stride=1, padding=0, bias=是否使用偏差),
            层归一化(过滤器数量 * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(过滤器数量 * 2, 1, kernel_size=1, stride=1, padding=0, bias=是否使用偏差)
        ]

        self.网络 = nn.Sequential(*self.网络)

    def forward(self, 输入):
        return self.网络(输入)
