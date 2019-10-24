import numpy as np 

m = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(m)
# displays the following result:
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])

print(m.T)
# displays the following result:
# array([[ 1,  5,  9],
#        [ 2,  6, 10],
#        [ 3,  7, 11],
#        [ 4,  8, 12]])


m_t = m.T
m_t[3][1] = 200
print(m_t)
# displays the following result:
# array([[ 1,   5, 9],
#        [ 2,   6, 10],
#        [ 3,   7, 11],
#        [ 4, 200, 12]])

print(m)
# displays the following result:
# array([[ 1,  2,  3,   4],
#        [ 5,  6,  7, 200],
#        [ 9, 10, 11,  12]])

inputs = np.array([[-0.27,  0.45,  0.64, 0.31]])
print(inputs)
# displays the following result:
# array([[-0.27,  0.45,  0.64,  0.31]])

print(inputs.shape)
# displays the following result:
# (1, 4)

weights = np.array([[0.02, 0.001, -0.03, 0.036], \
    [0.04, -0.003, 0.025, 0.009], [0.012, -0.045, 0.28, -0.067]])

print(weights)
# displays the following result:
# array([[ 0.02 ,  0.001, -0.03 ,  0.036],
#        [ 0.04 , -0.003,  0.025,  0.009],
#        [ 0.012, -0.045,  0.28 , -0.067]])

print(weights.shape)
# displays the following result:
# (3, 4)

#print(np.matmul(inputs, weights))
# displays the following error:
# ValueError: shapes (1,4) and (3,4) not aligned: 4 (dim 1) != 3 (dim 0)

print(np.matmul(inputs, weights.T))
# displays the following result:
# array([[-0.01299,  0.00664,  0.13494]])

print(np.matmul(weights, inputs.T))
# displays the following result:
# array([[-0.01299],# 
#        [ 0.00664],
#        [ 0.13494]])