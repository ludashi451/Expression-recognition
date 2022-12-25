import torch
import numpy as np

# # Convert tensor to numpy
# a = torch.ones(3)
# b = a.numpy()
# print(a, b)
# a += 1
# print(a, b)
#
# # Convert numpy to tensor
# c = np.ones(3)
# d = torch.from_numpy(c)
# print(c, d)
# c += 1
# print(c, d)


a = [1, 2, 3, 5, 4, 6, 10, 7]
b = np.array([[1, 2, 3, 5], [4, 6, 2, 6]])
print(a.index(min(a)))  # 返回第一个最小值的位置
print(a.index(max(a)))  # 返回第一个最大值的位置
print(np.max(b))  # 返回最大值
print(np.min(b))  # 返回最小值
print(np.argmax(b))  # 返回第一个最小值的位置
print(np.argmin(b))  # 返回第一个最大值的位置
