## This is course material for Introduction to Python Scientific Programming
## Example code: matplotlib_sigmoid.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import numpy as np
import matplotlib.pyplot as plt


x = np.arange(1., 2.,0.001)
x1 = np.arange(2.,3.,0.001)

y0 = 2*x
y1 = -3*x1+10

plt.plot(x, y0, 'b', linewidth = 2)
plt.plot(x1, y1, 'b', linewidth = 2)

plt.xlim(1.0, 3.0)
plt.ylim(1.0, 4.0)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Sample graph!")
plt.xticks(np.arange(1., 3.5, 0.5))
plt.yticks(np.arange(1., 4.5, 0.5))


plt.show()


