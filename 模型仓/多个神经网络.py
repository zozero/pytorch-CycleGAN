from torch.optim import lr_scheduler


def 获取调度器(优化器, 选项):
    if 选项.学习率策略 == '线性':
        def lambda_规则(轮回):
            # 这个公式什么意思暂时没看懂......
            线性学习率 = 1.0 - max(0, 轮回 + 选项.轮回起始数 - 选项.轮回次数) / float(选项.轮回衰减数 + 1)
            return 线性学习率
        调度器 = lr_scheduler.LambdaLR(优化器, lr_lambda=lambda_规则)
    elif 选项.学习率策略 == 'step':
        调度器 = lr_scheduler.StepLR(优化器, step_size=选项.迭代衰减间隔, gamma=0.1)
    elif 选项.学习率策略 == 'plateau':
        调度器 = lr_scheduler.ReduceLROnPlateau(优化器, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif 选项.学习率策略 == 'cosine':
        调度器 = lr_scheduler.CosineAnnealingLR(优化器, T_max=选项.轮回次数, eta_min=0)
    else:
        return NotImplementedError("学习率策略[%s]没被实施", 选项.学习率策略)

    return 调度器
