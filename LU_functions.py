import numpy as np

def lu_pp(A):
    n = len(A)-1
    P = np.linspace(0,n,n+1,dtype=int)
    for k in range(n):
        max_index = np.argmax(abs(A[k:,k]),axis=0)                      # Gathering row of max index w.r.t. k
        max_index = max_index + k                                       # Globalizing row index
        A[[k, max_index],:] = A[[max_index, k],:]                       # Permuting A matrix
        P[[k, max_index]]   = P[[max_index, k]]                         # Permuting vector
        gauss_w = np.reshape(A[k + 1:, k]/A[k, k],(n-k,1))              # Computing Gaussian Weights
        A[k + 1:, k:] = A[k + 1:, k:] - np.multiply(gauss_w,A[k, k:])   # Gaussian elimination
        A[k + 1:, k]  = np.squeeze(gauss_w)                             # Substituting GW in L part of A matrix
    return A, P

def lt_solve(A,b,P):
    n = len(A)-1
    b = b[P]
    A = np.tril(A,-1) + np.eye(n+1)                    # Selecting lower triangular values of A (w/ Gaussian weights)
    for i in range(1,n+1):
        b[i] = b[i] - np.matmul(A[i,:i],b[:i])         # Solving lower triangular
    return b

def ut_solve(A,b):
    n = len(A)-1
    A = np.triu(A)                          # Selecting upper triangular values of A
    b[n] = b[n]/A[n,n]                      # Computing last entry
    for i in range(n-1,-1,-1):
        b[i] = b[i] - np.matmul(A[i,i+1:],b[i+1:])   # Solving upper triangular
        b[i] = b[i]/A[i,i]                           # Dividing by diagonal entry
    return b


A, P = lu_pp(A)
b = lt_solve(A,b,P)
b = ut_solve(A,b)
print('this is b')
print(b)