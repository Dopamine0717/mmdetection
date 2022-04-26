# 测试一下Cross Entropy Loss的weight参数
import torch
import torch.nn as nn

batch_size = 10
nb_classes = 2

model = nn.Linear(10, nb_classes)
weight = torch.empty(nb_classes).uniform_(0, 1)
# 初始化CrossEntropy函数时传入各个class的权重, 
# 且设置reduction为None表示不进行聚合，返回一个loss数组
criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')

# This would be returned from your DataLoader
x = torch.randn(batch_size, 10)
target = torch.empty(batch_size, dtype=torch.long).random_(nb_classes)
sample_weight = torch.empty(batch_size).uniform_(0, 1)

output = model(x)
loss = criterion(output, target)
print(loss, loss.mean())
# 各个样本乘以其权重，然后求均值
loss = loss * sample_weight
print(loss, loss.mean())
loss.mean().backward()