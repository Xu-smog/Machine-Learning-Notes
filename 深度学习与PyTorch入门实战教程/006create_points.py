import numpy as np
from matplotlib import pyplot as plt 
#print(np.__version__)
#print(np.random.randn())

w = 1.347
b = 0.627

x = np.random.randn(300, 1) * 10
#print(x)
noize = np.random.randn(300, 1) * 7

y = w * x + b + noize
#print(y)

plt.title('sample points')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y, 'ob')
plt.show()

pArray = np.hstack((x, y))
#print(pArray)

np.savetxt('points.csv', pArray, delimiter=',')