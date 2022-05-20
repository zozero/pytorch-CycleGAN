from 选项仓.基础选项类 import 基础选项


class 测试用选项(基础选项):
    def 进行初始化(self, 解析器):
        解析器 = 基础选项.进行初始化(self, 解析器)
        解析器.add_argument('--结果目录', type=str, default='./检查点仓/', help='结果保存在这里')
        解析器.add_argument('--伸缩比例', type=float, default=1.0, help='结果图片伸缩比例')
        解析器.add_argument('--阶段', type=str, default='test', help='训练，验证，测试，其他等等')

        解析器.add_argument('--是否使用评估模式', action='store_true', help='在测试期间是否使用评估模式')
        解析器.add_argument('--测试图片数', type=int, default=50, help='片在测试时使用多少张图片')

        解析器.set_defaults(模式='测试')
        解析器.set_defaults(载入后尺寸=解析器.get_default('裁剪后尺寸'))

        self.是否为训练模式 = False
        return 解析器
