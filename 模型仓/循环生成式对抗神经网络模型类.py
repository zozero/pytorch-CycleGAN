from 模型仓 import 多个神经网络
from 模型仓.基础模型类 import 基础模型


class 循环生成式对抗神经网络模型(基础模型):
    def __init__(self, 选项):
        基础模型.__init__(self, 选项)
        # 指定要打印的训练产生的损失值。“个性”是指 结果B->生成器A到B->结果B
        self.损失值名称列表 = ['判别器A', '生成器A', '循环一致性A', '个性A', '判别器B', '生成器B', '循环一致性B', '个性B']
        # 指定要保存/显示的图片。（重建 reconstruction images  重新构建的图片）
        可视化图片名列表A = ['真A', '假B', '重建A']
        可视化图片名列表B = ['真B', '假A', '重建B']
        if self.是否为训练模式 and self.选项.lambda_个性 > 0.0:
            可视化图片名列表A.append('个性B')
            可视化图片名列表B.append('个性A')

        self.可视化图片名列表 = 可视化图片名列表A + 可视化图片名列表B
        if self.是否为训练模式:
            self.模型名称列表 = ['生成器A', '生成器B', '判别器A', '判别器B']
        else:  # 测试时只需生成器
            self.模型名称列表 = ['生成器A', '生成器B']
        # 定义网络
        self.生成器网络A = 多个神经网络.定义生成器(选项.输入的通道数, 选项.输出的通道数, 选项.生成器末尾过滤器数量, 选项.生成器模型类型, 选项.归一化类型, not 选项.无失活率, 选项.网络初始化类型,
                                   选项.初始化比例因子, self.图形处理单元标识码)
        self.生成器网络B = 多个神经网络.定义生成器(选项.输入的通道数, 选项.输出的通道数, 选项.生成器末尾过滤器数量, 选项.生成器模型类型, 选项.归一化类型, not 选项.无失活率, 选项.网络初始化类型,
                                   选项.初始化比例因子, self.图形处理单元标识码)

        if self.是否为训练模式:
            self.判别器网络A=多个神经网络.定义判别器()

    @staticmethod
    def 修改命令行选项(解析器, 是否为训练模式=True):
        解析器.set_defaults(no_dropout=True)
        if 是否为训练模式:
            解析器.add_argument('--lambda_A', type=float, default=10.0, help='循环损失权重 (A -> B -> A)')
            解析器.add_argument('--lambda_B', type=float, default=10.0, help='循环损失权重 (B -> A -> B)')
            解析器.add_argument('--lambda_个性', type=float, default=0.5,
                             help='使用个性映射。 将 lambda_个性 设置为 0 以外的值具有缩放个性映射损失权重的效果。 ' +
                                  '例如，如果个性损失的权重应该比重建损失的权重小 10 倍，请设置 lambda_个性 = 0.1')
        return 解析器
