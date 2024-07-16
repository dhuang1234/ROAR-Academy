import numpy as np


def swap_matrix_rows(M,a,b):
    if (a>=M.shape[0] or b>=M.shape[0] or a<0 or b<0): 
        raise IndexError("invalid rows")
    M[[a,b]] = M[[b,a]]
    return M


def swap_matrix_columns(M,a,b):
    if (a>=M.shape[1] or b>=M.shape[1] or a<0 or b<0): 
        raise IndexError("invalid columns")
    M[:,[a,b]] = M[:,[b,a]]
    return M

arr = np.array([[0,1,2,3,4,5],[10,11,12,13,14,15],[20,21,22,23,24,25],[30,31,32,33,34,35],[40,41,42,43,44,45],[50,51,52,53,54,55]])

print("columns swapped")
print(swap_matrix_columns(arr,0,3))
print("rows swapped")
print(swap_matrix_rows(arr,0,3))