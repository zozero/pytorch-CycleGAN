from 模型仓.基础模型类 import 基础模型


class 循环生成式对抗神经网络模型(基础模型):
    def __init__(self, 选项):
        super().__init__(选项)

    @staticmethod
    def 修改命令行选项(选项, 是否为训练模式=True):
        选项.set_defaults(no_dropout=True)
        if 是否为训练模式:
            选项.add_argument('--lambda_A', type=float, default=10.0, help='循环损失权重 (A -> B -> A)')
            选项.add_argument('--lambda_B', type=float, default=10.0, help='循环损失权重 (B -> A -> B)')
            选项.add_argument('--lambda_identity', type=float, default=0.5,
                            help='使用特性映射。 将 lambda_identity 设置为 0 以外的值具有缩放特性映射损失权重的效果。 ' +
                                 '例如，如果特性损失的权重应该比重建损失的权重小 10 倍，请设置 lambda_identity = 0.1')  # 有待进一步确定意义
        return 选项
