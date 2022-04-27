from 参数仓.基础参数类 import 基础参数


class 训练用参数(基础参数):
    def 进行初始化(self, 解析器):
        解析器=基础参数.进行初始化(self, 解析器)
        # 可视化控制类和超文本标记语言显示时的解析器
        解析器.add_argument('--显示的频率', type=int, default=400, help='在屏幕上显示训练结果的频率')
        解析器.add_argument('--显示的行数', type=int, default=4, help='如果是正数，则在可视化控制类（visdom：Visual dominate）的单个网页面板中显示所有图像，每行有一定数量的图像。')
        解析器.add_argument('--显示的标识', type=int, default=1, help='网页显示窗口的数字标识')
        解析器.add_argument('--显示的地址', type=str, default="http://localhost", help='可视化控制网页服务器的地址')
        解析器.add_argument('--显示的环境名', type=str, default="main", help='可视化控制类显示环境名称（默认为main）')
        解析器.add_argument('--显示的端口', type=int, default=8097, help='可视化控制网页服务器的端口')
        解析器.add_argument('--保存超文本标记语言频率', type=int, default=1000, help='将训练结果保存成超文本标记语言的频率')
        解析器.add_argument('--控制台更新频率', type=int, default=100, help='显示结果到控制台的频率')
        解析器.add_argument('--无超文本标记语言', action='store_true', help='不保存中间的训练结果到 [解析器.检查点目录]/[解析器.名称]/web/')
        # 网络保存和载入的解析器
        解析器.add_argument('--保存频率', type=int, default=5000, help='最新结果保存频率')
        解析器.add_argument('--保存迭代频率', type=int, default=5000, help='在每次迭代结束时检查点保存的频率')
        解析器.add_argument('--是否通过迭代来保存', action='store_true', help='是否通过迭代来保存模型')
        解析器.add_argument('--是否继续训练', action='store_true', help='继续训练会加载最新模型')
        解析器.add_argument('--起始迭代位置',  type=int, default=1,help='起始迭代位置，通过我们保存的模型（起始迭代位置，起始迭代位置+保存频率，......）')
        解析器.add_argument('--阶段',  type=str, default='训练',help='训练，验证，测试，等等')
        # 训练时参数
        解析器.add_argument('--迭代次数',  type=int, default=100,help='具有初始学习率的迭代数')
        解析器.add_argument('--迭代衰减次数',  type=int, default=100,help='将学习率线性衰减为零的迭代次数')
        解析器.add_argument('--动量项1',  type=float, default=0.5,help='片刻自适应估计算法的动量项')
        解析器.add_argument('--学习率',  type=float, default=0.0002,help='初始化片刻自适应估计算法的学习率')
        解析器.add_argument('--生成式对抗神经网络模式',  type=str, default='lsgan',help='生成式对抗神经网络目标的类型[vanilla| lsgan | wgangp]。香草（vanilla）生成式对抗网络损失函数是交叉熵目标函数，生成式对抗网络原始论文里有用到它')
        解析器.add_argument('--缓冲区图像尺寸',  type=int, default=50,help='存储先前生成的图像，在缓冲区图像的尺寸')
        解析器.add_argument('--学习率策略',  type=str, default='线性',help='学习率变化策略[线性 | step | plateau | cosine]')
        解析器.add_argument('--衰减间隔',  type=int, default=50,help='多少次迭代乘以一个伽马值，即多少次迭代衰减一次')

        self.是否为训练模式=True
        return 解析器