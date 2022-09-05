import numpy as np
a = np.zeros((2,2,3))
b = np.zeros((2,2,3))

c = np.concatenate(a)
print(c.shape)