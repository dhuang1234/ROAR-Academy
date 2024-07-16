import numpy as np

# Question 1
v = np.array([2.,2.,4.])

m1 = np.array([1.,0.,0.])
m2 = np.array([0.,1.,0.])
m3 = np.array([0.,0.,1.])

e0 = v@m1
e1 = v@m2
e2 = v@m3

print(e0)
print(e1)
print(e2)


