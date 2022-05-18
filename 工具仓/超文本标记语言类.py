import os.path

import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br


class 超文本标记语言:
    def __init__(self, 网站目录, 标题, 刷新=0):
        self.标题 = 标题
        self.网站目录 = 网站目录
        self.图片目录 = os.path.join(self.网站目录, '图片库')
        if not os.path.exists(self.网站目录):
            os.makedirs(self.网站目录)
        if not os.path.exists(self.图片目录):
            os.makedirs(self.图片目录)

        self.文档对象 = dominate.document(title=标题)
        if 刷新 > 0:
            with self.文档对象.head:
                meta(http_equiv='refresh', content=str(刷新))

    def 获得图片目录(self):
        return self.图片目录

    def 添加页眉(self, 文本):
        with self.文档对象:
            h3(文本)

    def 添加复数图片(self, 复数图片, 复数文本, 复数链接, 宽度=400):
        self.表格 = table(border=1, style="table-layout: fixed;")
        self.文档对象.add(self.表格)
        with self.表格:
            with tr():
                for 图片, 文本, 链接 in zip(复数图片, 复数文本, 复数链接):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('图片列表', 链接)):
                                img(style="width:%dpx" % 宽度, src=os.path.join('images', 图片))
                            br()
                            p(文本)

    def 保存(self):
        网页文件 = '%s/index.html' % self.网站目录
        文件 = open(网页文件, 'wt')
        文件.write(self.文档对象.render())
        文件.close()


if __name__ == '__main__':  # 我们在这里展示一个示例用法。
    超文本标记语言实例 = 超文本标记语言('网站/', '测试')
    超文本标记语言实例.添加页眉('你好世界')

    复数图片, 复数文本, 复数链接 = [], [], []
    for 计数 in range(4):
        复数图片.append('图片%d.png' % 计数)
        复数文本.append('文本%d' % 计数)
        复数链接.append('链接%d' % 计数)

    超文本标记语言实例.添加复数图片(复数图片, 复数文本, 复数链接)
    超文本标记语言实例.保存()
