from cgi import print_environ
import numpy as np
import torch

x = torch.tensor([[0.99,0.1,0.0,0.0,0.0]])
target = torch.tensor([0])
loss = torch.nn.CrossEntropyLoss()(x, target)
print(loss)