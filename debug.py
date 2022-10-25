from cgi import print_environ
import numpy as np
import torch

loss = torch.nn.MSELoss(reduction='sum')
input = torch.rand(3, 5, requires_grad=True)
target = torch.rand(3, 5)
output = loss(input, target)
print(loss)