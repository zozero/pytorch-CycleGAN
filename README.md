### 写在前面
原项目地址： [链接](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)<br>
数据集地址： [链接](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/)<br>

### 说明
该项目只实现了**CycleGAN（循环生成式对抗神经网络）**<br>
你可能需要在项目根目录下添加文件夹：检查点仓、数据仓
可能会用到其他很多的类库，需要下载<br>
可以在控制台执行命令 visdom 使用网页查看结果<br>
添加数据的参数，例如：--数据根目录 ./数据仓/horse2zebra<br>

### 命名
迭代数是指交换替代。这里就是样本的数量，每个样本循环一次神经网络<br>
轮回数是指佛教用语。因果报应的一种说教。这里就是所有样本都被迭代过一次的后，再从第一个样本重新循环的次数<br>
命名时为了能够易懂所以部分命名过长<br>

### 写在后面
现在的我才发现，语文的重要性......<br>
不是语言学家的程序员，不是一个好程序员<br>
一个很好用的pycharm插件 ChinesePinyinCodeCompletionHelper [链接](https://github.com/tuchg/ChinesePinyin-CodeCompletionHelper)<br>