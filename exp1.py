import numpy as np

n = int(input("Enter the dimension of matrices A and B and vector x: "))

# Input elements of A and B

print("Enter elements of matrix A row-wise:")
A = np.array([list(map(int, input().split())) for _ in range(n)])

print("Enter elements of matrix B row-wise:")
B = np.array([list(map(int, input().split())) for _ in range(n)])

print("Enter elements of column vector x:")
x = []
for i in range(n):
    temp = int(input())
    x.append(temp)
x = np.array(x)

print("A: ", A)
print("B:", B)

# Transpose

Atrans = np.transpose(A)
Btrans = np.transpose(B)
print("A transpose: ", Atrans)
print("B transpose: ", Btrans)

# A.B

print("Matrix multiplication AB is: ", np.matmul(A, B))

X = np.matmul(A, B)
X = np.transpose(X)
Y = np.matmul(Btrans, Atrans)

if np.array_equal(X, Y):
    print("(AB)T = BT AT verified")

detA = np.linalg.det(A)
detB = np.linalg.det(B)
print("Determinant of A: ", detA)
print("Determinant of B: ", detB)

if detA != 0:
    Ainv = np.linalg.inv(A)
    print("Inverse of A: ", Ainv)
else:
    print("AA^-1", np.matmul(A, Ainv))
    print("A is Singular")

if detB != 0:
    Binv = np.linalg.inv(B)
    print("Inverse of B: ", Binv)
    print("BB^-1", np.matmul(B, Binv))
else:
    print("B is Singular")

traceA = np.trace(A)
traceB = np.trace(B)
print("Trace of matrix A is: ", traceA)
print("Trace of matrix B is: ", traceB)

if np.array_equal(X, np.trace(np.matmul(B, A))):
    print("trace (AB)=trace (BA)")

ATB = np.trace(np.matmul(Atrans, B))
ABT = np.trace(np.matmul(A, Btrans))
BAT = np.trace(np.matmul(B, Atrans))
BTA = np.trace(np.matmul(Btrans, A))

if np.array_equal(ATB, ABT):
    if np.array_equal(ABT, BAT):
        if np.array_equal(BAT, BTA):
            print("trace(ATB) = trace(ABT) = trace (BAT) = trace(BTA)")

y = np.matmul(A, x)
print("Given y we find x so x is:")
print(np.matmul(y, Ainv))

dotxy = np.dot(x, y)
print("Inner product of x and y is:",)

if dotxy == 0:
    print("x and y are Orthogonal")

normx = np.linalg.norm(x)
normy = np.linalg.norm(y)
print("Norm of x:", normx)
print("Norm of y: ", normy)
print("Normalized x is:", x/normx)
print("Normalized y is:", y/normy)

# Cauchy-Schwarz inequality

lhs = np.linalg.norm(np.dot(x, y))
rhs = np.multiply(np.linalg.norm(x), np.linalg.norm(y))

if lhs <= rhs:
    print("x and y satisfy Cauchy-Schwarz inequality")

xty = np.matmul(np.transpose(x), y)
ytx = np.matmul(np.transpose(y), x)

if np.array_equal(xty, ytx):
    print("Verified xTy = yTx")

ytA = np.matmul(np.transpose(y), A)
ytAx = np.matmul(ytA, x)

xtAt = np.matmul(np.transpose(x), np.transpose(A))
xtAty = np.matmul(xtAt, y)

if np.array_equal(ytAx, xtAty):
    print("Verified yTAx = xTATY")

# Eigen values and Eigen vectors

def evd(matrix):
    eigenvalue, eigenvector = np.linalg.eig(matrix)
    print("Eigen value:", eigenvalue)
    print("Eigen vector: ", eigenvector)

    U = eigenvector
    A = np.diag(eigenvalue)
    U_inv = np.linalg.inv(U)
    matrix_decomposed = np.dot(np.dot(U, A), U_inv)

    if np.array_equal(matrix_decomposed, matrix):
        print("EVD verified")

evd(A)
evd(B)
