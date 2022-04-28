import os.path

图片扩展名列表 = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]


def 判断是否为图片文件(文件名):
    return any(文件名.endswith(扩展名) for 扩展名 in 图片扩展名列表)


def 制作数据集(目录名, 数据集最大长度=float("inf")):
    图片列表 = []
    assert os.path.isdir(目录名), '%s不是有效的目录' % 目录名

    for 目录路径列表, _, 文件名列表 in sorted(os.walk(目录名)):
        for 文件名 in 文件名列表:
            if 判断是否为图片文件(文件名):
                路径 = os.path.join(目录路径列表, 文件名)
                图片列表.append(路径)
    return 图片列表[:min(数据集最大长度, len(图片列表))]
