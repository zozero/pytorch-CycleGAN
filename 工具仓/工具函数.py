import os.path


def 新建多个文件夹(路径列表):
    if isinstance(路径列表, list) and not isinstance(路径列表, str):
        for 路径 in 路径列表:
            新建文件夹(路径)
    else:
        新建文件夹(路径列表)


def 新建文件夹(路径):
    if not os.path.exists(路径):
        os.makedirs(路径)
