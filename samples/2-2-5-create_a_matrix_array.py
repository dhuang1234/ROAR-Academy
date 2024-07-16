import numpy as np

def set_array(L,rows, cols, order= "row"):
    if (order=="column"):
        return np.array(L).transpose()
    return np.array(L)



arr = [[0,1,2,3,4,5],[10,11,12,13,14,15],[20,21,22,23,24,25],[30,31,32,33,34,35],[40,41,42,43,44,45],[50,51,52,53,54,55]]

print(set_array(arr,6,6,"column"))