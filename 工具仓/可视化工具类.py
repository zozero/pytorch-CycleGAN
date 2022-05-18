import os.path
import sys
import time
from subprocess import Popen, PIPE

import numpy as np

from . import 超文本标记语言类, 工具函数

try:
    import wandb
except ImportError:
    import wandb

    print('警告：wandb包没有找到。选项“是否使用权重和偏置项数据库”的结果是错误的')

if sys.version_info[0] == 2:
    可视化工具异常基础 = Exception
else:
    可视化工具异常基础 = ConnectionError


class 可视化工具:
    def __init__(self, 选项):
        self.选项 = 选项
        self.显示的标识 = 选项.显示的标识
        self.是否使用超文本标记语言 = 选项.是否为训练模式 and not 选项.无超文本标记语言
        self.窗口尺寸 = 选项.窗口尺寸
        self.名称 = 选项.名称
        self.端口 = 选项.服务器的端口
        self.是否已保存 = False
        self.是否使用数据库 = 选项.是否使用权重和偏置项数据库  # wandb
        self.当前轮回 = 0
        self.显示的行数 = 选项.显示的行数

        if self.显示的标识 > 0:
            import visdom
            # 这里需要先在控制台运行 visdom 参考地址：https://github.com/fossasia/visdom#usage
            self.可视化实例 = visdom.Visdom(server=选项.服务器的地址, port=选项.服务器的端口, env=选项.可视化工具的环境名)
            if not self.可视化实例.check_connection():
                self.创建可视化实例连接()
        if self.是否使用数据库:
            # 需要自己注册，要先登录
            project = '循环生成式对抗神经网络和像素到像素神经网络'
            # project = 'my-test-project'
            self.数据库运行实例 = wandb.init(project=project, name=选项.名称, entity="zozero") if not wandb.run else wandb.run
            self.数据库运行实例.config = {
                "learning_rate": 0.0002,
                "epochs": 100,
                "batch_size": 128
            }
            self.数据库运行实例._label(repo='循环生成式对抗神经网络和像素到像素神经网络')
        if self.是否使用超文本标记语言:
            self.网站文件目录 = os.path.join(选项.检查点目录, 选项.名称, '网站')
            self.图片目录 = os.path.join(self.网站文件目录, '图片库')
            print('创建网站目录 %s ......' % self.网站文件目录)
            工具函数.新建多个文件夹([self.网站文件目录, self.图片目录])
            self.日志文件路径 = os.path.join(选项.检查点目录, 选项.名称, '损失值日志.txt')
            with open(self.日志文件路径, 'a') as 日志文件:
                现在 = time.strftime("%c")
                日志文件.write('================ 训练时损失值 (%s) ================\n' % 现在)

    def 重置(self):
        self.是否已保存 = False

    def 创建可视化实例连接(self):
        命令行命令 = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.端口
        print('\n\n不能连接可视化工具的服务。\n 尝试在命令行中开始服务......')
        print('命令行命令：%s' % 命令行命令)
        # stdout 标准输出   stderr 标准错误
        Popen(命令行命令, shell=True, stdout=PIPE, stderr=PIPE)

    def 显示当前结果(self, 视觉效果字典, 轮回索引, 是否保存结果):
        """
        :param 视觉效果字典: 要显示或保存的图像字典
        :param 轮回索引:
        :param 是否保存结果: 是否使用 visdom 在浏览器中显示图像
        :return:
        """
        if self.显示的标识 > 0:
            行数 = self.显示的行数
            if 行数 > 0:
                行数 = min(行数, len(视觉效果字典))
                h, w = next(iter(视觉效果字典.values())).shape[:2]
                样式表 = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                标题 = self.名称
                超文本标记语言标签 = ''
                超文本标记语言列标签 = ''
                图片列表 = []
                索引 = 0
                for 标签, 图片 in 视觉效果字典.items():
                    图片矩阵 = 工具函数.把张量转成图片(图片)
                    超文本标记语言列标签 += '<td>%s</td>' % 标签
                    图片列表.append(图片矩阵.transpose([2, 0, 1]))
                    索引 += 1
                    if 索引 % 行数 == 0:
                        超文本标记语言标签 += '<tr>%s</tr>' % 超文本标记语言列标签
                        超文本标记语言列标签 = ''
                白色的图片 = np.ones_like(图片矩阵.transpose([2, 0, 1])) * 255
                while 索引 % 行数 != 0:
                    图片列表.append(白色的图片)
                    超文本标记语言列标签 += '<td></td>'
                    索引 += 1
                if 超文本标记语言列标签 != '':
                    超文本标记语言标签 += '<tr>%s</tr>' % 超文本标记语言列标签
                try:
                    self.可视化实例.images(图片列表, nrow=行数, win=self.显示的标识 + 1, padding=2, opts=dict(title=标题 + '图片列表'))
                    超文本标记语言标签 = '<table>%s</table>' % 超文本标记语言标签
                    self.可视化实例.text(样式表 + 超文本标记语言标签, win=self.显示的标识 + 2, opts=dict(title=标题 + '标签'))

                except 可视化工具异常基础:
                    self.创建可视化实例连接()
            else:
                索引 = 1
                try:
                    for 标签, 图片 in 视觉效果字典.items():
                        图片矩阵 = 工具函数.把张量转成图片(图片)
                        self.可视化实例.images(图片矩阵.transpose([2, 0, 1]), win=self.显示的标识 + 索引, opts=dict(title=标签))
                        索引 += 1
                except 可视化工具异常基础:
                    self.创建可视化实例连接()

            if self.是否使用数据库:
                列 = [键值 for 键值, _ in 视觉效果字典.items()]
                列.insert(0, '轮回')
                结果表 = wandb.Table(columns=列)
                表格行 = [轮回索引]
                图片字典 = {}
                for 标签, 图片 in 视觉效果字典.items():
                    图片矩阵 = 工具函数.把张量转成图片(图片)
                    数据库图片 = wandb.Image(图片矩阵)
                    表格行.append(数据库图片)
                    图片字典[标签] = 数据库图片
                self.数据库运行实例.log(图片字典)
                if 轮回索引 != self.当前轮回:
                    self.当前轮回 = 轮回索引
                    结果表.add_data(*表格行)
                    self.数据库运行实例.log({"结果": 结果表})

            if self.是否使用超文本标记语言 and (是否保存结果 or not self.是否已保存):
                self.是否已保存 = True
                for 标签, 图片 in 视觉效果字典.items():
                    图片矩阵 = 工具函数.把张量转成图片(图片)
                    图片路径 = os.path.join(self.图片目录, '轮回%.3d_%s.png' % (轮回索引, 标签))
                    工具函数.保存图片(图片矩阵, 图片路径)

                # 更新站点
                网页 = 超文本标记语言类.超文本标记语言(self.网站文件目录, '实验名称 = %s' % self.名称, 刷新=1)
                for 计数 in range(轮回索引, 0, -1):
                    网页.添加页眉('轮回次数 [%d]' % 计数)
                    图片列表, 文本列表, 链接列表 = [], [], []

                    for 标签, 图片 in 视觉效果字典.items():
                        # 图片矩阵 = 工具函数.把张量转成图片(图片)
                        图片路径 = '轮回%.3d_%s.png' % (计数, 标签)
                        图片列表.append(图片路径)
                        文本列表.append(标签)
                        链接列表.append(图片路径)
                    网页.添加复数图片(图片列表, 文本列表, 链接列表, 宽度=self.窗口尺寸)
                网页.保存()

    def 统计当前的损失值(self, 轮回索引, 轮回进度, 损失值字典):
        """

        :param 轮回索引:
        :param 轮回进度: 当前 轮回 的进度（百分比），介于 0 到 1 之间
        :param 损失值字典:
        :return:
        """
        if not hasattr(self, '统计的数据'):
            self.统计的数据 = {'X': [], 'Y': [], '标目': list(损失值字典.keys())}
        self.统计的数据['X'].append(轮回索引 + 轮回进度)
        self.统计的数据['Y'].append([损失值字典[k] for k in self.统计的数据['标目']])
        try:
            self.可视化实例.line(
                X=np.stack([np.array(self.统计的数据['X'])] * len(self.统计的数据['标目']), 1),
                Y=np.stack(self.统计的数据['Y']),
                opts={
                    'title': self.名称 + ' 损失值变化趋势图',
                    'legend': self.统计的数据['标目'],
                    'xlabel': '轮回',
                    'ylabel': '损失值'
                },
                win=self.显示的标识
            )
        except 可视化工具异常基础:
            self.创建可视化实例连接()
        if self.是否使用数据库:
            self.数据库运行实例.log(损失值字典)

    def 显示当前损失值(self, 轮回索引, 单次轮回的迭代数, 损失值字典, 平均时间, 时间记录):
        """

        :param 轮回索引:
        :param 单次轮回的迭代数:
        :param 损失值字典:
        :param 平均时间:
        :param 时间记录: 每批的数据加载时间（由 每批数量 平均）
        :return:
        """
        消息 = '(轮回索引：%d，迭代次数：%d，时间：%.3f，数据：%.3f)\t' % (轮回索引, 单次轮回的迭代数, 平均时间, 时间记录)
        for k, v in 损失值字典.items():
            消息 += '%s：%.3f\t' % (k, v)
        print(消息)
        with open(self.日志文件路径, "a") as 日志:
            日志.write('%s\n' % 消息)
