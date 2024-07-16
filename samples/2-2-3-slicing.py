import numpy as np

arr = np.array([[0,1,2,3,4,5],[10,11,12,13,14,15],[20,21,22,23,24,25],[30,31,32,33,34,35],[40,41,42,43,44,45],[50,51,52,53,54,55]])

#blue
print(arr[0:6,1])
#pink
print(arr[1,2:4])
#orange 
print(arr[2:6:2,0:6:2])
#green
print(arr[2:4,4:6])

print(arr[0:1:])