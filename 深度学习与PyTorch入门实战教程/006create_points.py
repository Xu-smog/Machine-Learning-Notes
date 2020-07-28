import numpy as np
from matplotlib import pyplot as plt 
#print(np.__version__)
#print(np.random.randn())

w = 1.347
b = 0.627

#pArray = np.array(np.zeros((100, 2), dtype=float))
#print(pArray)
x = np.random.randint(30, 80, [100, 1])
print(x)
noize = np.random.randint(-5, 5, [100, 1])

y = w * x + b + noize
print(y)