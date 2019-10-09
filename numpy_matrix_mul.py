import numpy as np

m = np.array([[1,2,3],[4,5,6]])
print(m)
# displays the following result:
# array([[1, 2, 3],
#        [4, 5, 6]])

n = m * 0.25
print(n)
# displays the following result:
# array([[ 0.25,  0.5 ,  0.75],
#        [ 1.  ,  1.25,  1.5 ]])

print(m * n)
# displays the following result:
# array([[ 0.25,  1.  ,  2.25],
#        [ 4.  ,  6.25,  9.  ]])

print(np.multiply(m, n))   # equivalent to m * n
# displays the following result:
# array([[ 0.25,  1.  ,  2.25],
#        [ 4.  ,  6.25,  9.  ]])



a = np.array([[1,2,3,4],[5,6,7,8]])
print(a)
# displays the following result:
# array([[1, 2, 3, 4],
#        [5, 6, 7, 8]])
print(a.shape)
# displays the following result:
# (2, 4)

b = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(b)
# displays the following result:
# array([[ 1,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9],
#        [10, 11, 12]])
print(b.shape)
# displays the following result:
# (4, 3)

c = np.matmul(a, b)
print(c)
# displays the following result:
# array([[ 70,  80,  90],
#        [158, 184, 210]])
print(c.shape)
# displays the following result:
# (2, 3)


# np.matmul(b, a)
# displays the following error:
# ValueError: shapes (4,3) and (2,4) not aligned: 3 (dim 1) != 2 (dim 0)

a = np.array([[1,2],[3,4]])
print(a)
# displays the following result:
# array([[1, 2],
#        [3, 4]])

print(np.dot(a,a))
# displays the following result:
# array([[ 7, 10],
#        [15, 22]])

print(a.dot(a))  # you can call `dot` directly on the `ndarray`
# displays the following result:
# array([[ 7, 10],
#        [15, 22]])

print(np.matmul(a,a))
# array([[ 7, 10],
#        [15, 22]])