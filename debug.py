from cgi import print_environ
import numpy as np
import torch
# (L, N, num_agent, ...)
a = torch.tensor([1.,2.,3.,4.],requires_grad=True)
d = a
b = torch.tensor([6.,6.,6.,6.], requires_grad=True)

a = a*2

print(d)
print(a)