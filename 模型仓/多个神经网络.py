from torch.optim import lr_scheduler

def 获取调度器(优化器,选项):
    if 选项.学习率策略=='线性':
        def lambda_规则(迭代):
            pass