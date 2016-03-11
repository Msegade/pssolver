import sys
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spl
import scipy.io as sio

iters = 0

def solve_sparse(A,b):
    def callback(xk):
        global iters 
        iters+=1 
    return spl.cg(A,b,maxiter=1000,tol=1e-12, callback=callback)

b = np.loadtxt(sys.argv[2], skiprows=1);
inputfile =  (sys.argv[1])
# Read file into a scipy.sparse.coo.coo_matrix object
A = sio.mmread(inputfile)
# The solver only reads the matrix in csr
A = A.tocsr()

result = solve_sparse(A,b)
np.savetxt("result.txt", result[0])
print("Number of iterations = ", iters)

