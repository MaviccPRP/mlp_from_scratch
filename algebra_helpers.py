# scalar multiplication
def scalar_mult(k, X):
    res = []
    for i in X:
        row = []
        for j in i:
            row.append(k * j)
        res.append(row)
    return res



# matrix mutiplication
def dot(X,Y):
    if len(X[0]) != len(Y):
        return "Matrices do not match for multiplication"
    return [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]

# hadamard product
def hadamard(X,Y):
    res = []
    if not any(isinstance(el, list) for el in X):
        for i in Y:
            row = []
            for d, j in enumerate(X):
                row.append(i[d] * j)
            res.append(row)
    else:
        for i, j in zip(X,Y):
            row = []
            if isinstance(i, list):
                for k, l in zip(i,j):
                    row.append(k*l)
                res.append(row)

    return res

l1_delta = [0.017329524383939823, 0.029647815559742975, -0.0012236811810426591, -0.0050457655158837947, -0.036635254072932777]
relu = [[ 1.,  1.,  1.,  1.,  1.]]
res = [[  2.50386570e-04,   3.77547690e-04,  -1.55556644e-05,  -6.76090683e-05,
   -5.02015585e-04]]



# element wise summing up of two matrices
def summarize(X,Y):
    res = []
    for i, j in zip(X, Y):
        row = []
        for k, l in zip(i, j):
            row.append(k + l)
        res.append(row)
    return res

def substract(X,Y):
    res = []
    for i, j in zip(X, Y):
        row = []
        for k, l in zip(i, j):
            row.append(k - l)
        res.append(row)
    return res

def maximum(X):
    return max(X)

def transpose(X):
    return list(zip(*X))

# 4x4 matrix
X = [[1, 2, 3, 4],
[1, 2, 3, 4],
[1, 2, 3, 4],
[5, 4, 2, 1]]

X_2 = [[3, 2, 3, 4],
[3, 2, 3, 4],
[3, 2, 3, 4],
[3, 4, 2, 1]]

vec1 = [[1,2,3,4]]
vec2 = [[1,2,3,4]]

# 4x2 matrix
Y = [[3, 4],
[6, 1],
[3, 2],
[11, 2]]

def sum_matrix(X, axis=0):
    if axis == 0:
        return [sum(row[i] for row in X) for i in range(len(X[0]))]
    if axis == 1:
        rows = len(X)
        cols = len(X[0])
        total = []
        for x in range(0, rows):
            rowtotal = 0
            for y in range(0, cols):
                rowtotal = rowtotal + X[x][y]
            total.append(rowtotal)
        return total

