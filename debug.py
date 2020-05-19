import torch
# from main import custom_data1
from tqdm import tqdm
import time
import numpy as np

#
# y_hat = torch.tensor([[0,1], [1,0]]).long()
# y_true = torch.tensor([[1], [1]], dtype=torch.int64)
# res = torch.nn.functional.cross_entropy(y_hat, y_true)
# print(res)
#
target = torch.randint(5, (3,), dtype=torch.int64)
input = torch.randn(3, 5, requires_grad=True)
print(input, target)
# y_hat = torch.tensor([[0.2,0.8], [0.2,0.8]])
# y_true = torch.tensor([1,0]).long()
# res = torch.nn.functional.cross_entropy(y_hat, y_true)
# print(res)
# print(0.66/16)