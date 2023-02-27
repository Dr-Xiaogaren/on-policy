from cgi import print_environ
import numpy as np
import torch

x = torch.tensor([[0.99,0.1,0.5,0.8,0.9]])
y = torch.tensor([[True, True, False, True, False]])

loss = x*y
print(loss)