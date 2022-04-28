import torch.utils.data as data
from abc import ABC, abstractmethod
from PIL import Image
import torchvision.transforms as transforms


class 基础数据处理(data.Dataset, ABC):
    def __init__(self, 选项):
        self.选项 = 选项
        self.根数据根目录 = 选项.数据根目录

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, item):
        pass


def 进行转化(选项, 参数=None, 是否转换为灰度图=False, 转化方法=transforms.InterpolationMode.BICUBIC, 转换=True):
    转化列表 = []
    if 是否转换为灰度图:
        转化列表.append(transforms.Grayscale(1))

    if '重置' in 选项.图像预处理:
        图片尺寸 = [选项.载入后尺寸, 选项.载入后尺寸]
        转化列表.append(transforms.Resize(图片尺寸, 转化方法))
    elif 'scale_width' in 选项.图像预处理:
        转化列表.append(transforms.Lambda(lambda 图片: __按宽度与比例调整(图片, 选项.载入后尺寸, 选项.裁剪后尺寸, 转化方法)))

    if '裁剪' in 选项.图像预处理:
        if 参数 is None:
            转化列表.append(transforms.RandomCrop(选项.裁剪后尺寸))
        else:
            转化列表.append(transforms.Lambda(lambda 图片: __裁剪(图片, 参数['裁剪位置'], 选项.裁剪后尺寸)))

    if 选项.图像预处理 == 'none':
        转化列表.append(transforms.Lambda(lambda 图片: __调整为基数的整数倍2(图片, 基数=4, 调整方法=transforms.InterpolationMode.BICUBIC)))

    if not 选项.不翻转:
        if 参数 is None:
            转化列表.append(transforms.RandomHorizontalFlip())
        elif 参数['翻转']:
            转化列表.append(transforms.Lambda(lambda 图片: __翻转(图片, 参数['翻转'])))

    if 转换:
        转化列表 += [transforms.ToTensor()]
        if 是否转换为灰度图:
            转化列表 += [transforms.Normalize((0.5,), (0.5,))]
        else:
            转化列表 += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(转化列表)


def __调整为基数的整数倍2(图片, 基数, 调整方法=transforms.InterpolationMode.BICUBIC):
    图宽, 图高 = 图片.size
    宽 = int(round(图宽 / 基数) * 基数)
    高 = int(round(图高 / 基数) * 基数)
    if 宽 == 图宽 and 高 == 图高:
        return 图片

    __打印尺寸调整警告(图宽, 图高, 宽, 高)
    return 图片.resize((宽, 高), 调整方法)


def __按宽度与比例调整(图片, 目标尺寸, 裁剪尺寸, 调整方法=transforms.InterpolationMode.BICUBIC):
    图宽, 图高 = 图片.size
    if 图宽 == 目标尺寸 and 图高 == 裁剪尺寸:
        return 图片
    宽 = 目标尺寸
    高 = int(max(目标尺寸 * 图高 / 图宽, 裁剪尺寸))
    return 图片.resize((宽, 高), 调整方法)


def __裁剪(图片, 位置, 尺寸):
    图宽, 图高 = 图片.size
    x1, y1 = 位置
    宽 = 高 = 尺寸
    if 图宽 > 宽 or 图高 > 高:
        return 图片.crop((x1, y1, x1 + 宽, y1 + 高))
    return 图片


def __翻转(图片, 翻转):
    if 翻转:
        return 图片.transpose(Image.FLIP_LEFT_RIGHT)
    return 图片


def __打印尺寸调整警告(图宽, 图高, 宽, 高):
    if not hasattr(__打印尺寸调整警告, '已打印'):
        print(
            "图像大小需要是 4 的倍数。"
            "加载的图片大小为(%d, %d)，所以调整为(%d, %d)。"
            "将对所有尺寸不是 4 倍数图像进行此调整" % (图宽, 图高, 宽, 高)
        )
        __打印尺寸调整警告.已打印 = True
