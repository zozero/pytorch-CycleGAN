import os.path

from 工具仓 import 超文本标记语言类
from 工具仓.可视化工具类 import 保存复数图片
from 数据处理仓 import 创建数据集
from 模型仓 import 创建模型
from 选项仓.测试用选项类 import 测试用选项

try:
    import wandb
except ImportError:
    import wandb
    print('警告：找不到wandb包，请检查选项“--是否使用权重和偏置项数据库”结果是否错误。')

if __name__ == '__main__':
    选项 = 测试用选项().解析()
    选项.线程数 = 0
    选项.没批数量 = 1
    选项.是否按批拿取 = True
    选项.不翻转 = True
    选项.显示的标识 = -1
    数据集 = 创建数据集(选项)
    模型 = 创建模型(选项)
    模型.设置(选项)

    if 选项.是否使用权重和偏置项数据库:
        project = '循环生成式对抗神经网络和像素到像素神经网络'
        # project = 'my-test-project'
        数据库运行实例 = wandb.init(project=project, name=选项.名称, entity="zozero") if not wandb.run else wandb.run
        数据库运行实例.config = {
            "learning_rate": 0.0002,
            "epochs": 100,
            "batch_size": 128
        }
        数据库运行实例._label(repo='循环生成式对抗神经网络和像素到像素神经网络')

    网站目录 = os.path.join(选项.结果目录, 选项.名称, '{}_{}'.format(选项.阶段, 选项.轮回的位子))
    if 选项.迭代的位子 > 0:
        网站目录 = '{:s}_迭代{:d}'.format(网站目录, 选项.迭代的位子)
    print('创建网站目录：', 网站目录)
    网页 = 超文本标记语言类.超文本标记语言(网站目录, '实验名=%s，阶段=%s，轮回的位子=%s' % (选项.名称, 选项.阶段, 选项.轮回的位子))

    if 选项.是否使用评估模式:
        模型.评估()
    for 索引, 数据 in enumerate(数据集):
        if 索引 >= 选项.测试图片数:
            break
        模型.设置输入(数据)
        模型.测试()
        可视化 = 模型.获得当前视觉效果()
        图片路径 = 模型.取得图片路径()
        if 索引 % 5 == 0:
            print('进度：第（%04d）批\t图片：%s' % (索引, 图片路径))

        保存复数图片(网页, 可视化, 图片路径, 伸缩比例=选项.伸缩比例,宽度=选项.窗口尺寸,是否使用数据库=选项.是否使用权重和偏置项数据库)
    网页.保存()
