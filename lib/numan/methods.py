import numpy as np
import scipy.linalg
from numan import (
	matrix
)


"""
DIRETCT
"""

def cholesky(A, b):
	L = scipy.linalg.cholesky(A, lower=True)
	y = scipy.linalg.solve(L, b, lower=True)
	x = scipy.linalg.solve(L.T, y, lower=False)
	return x

def lu(A, b):
	lu,piv = scipy.linalg.lu_factor(A)
	x = scipy.linalg.lu_solve((lu, piv), b)
	return x

def svd(A,y):
	(U, s, VT) = scipy.linalg.svd(A)
	n = VT.shape[0]
	return ((U[:,:n].T @ y / s) @ VT).reshape((n,1))


"""
ITERATIVE
"""

def matIt_jacobi(A):
	E = -np.tril(A, k=-1)
	F = -np.triu(A, k=1)
	D = A + E + F
	M = D
	N = E + F
	J = np.linalg.inv(M) @ N
	return J

def matIt_gaussSiedel(A):
	E = -np.tril(A, k=-1)
	F = -np.triu(A, k=1)
	D = A + E + F
	M = np.subtract(D, E)
	N = F
	GS = np.linalg.inv(M) @ N
	return GS

"""
METHODS
"""

def jacobi(A, b, x0, maxit, tol, xTrue):
	n = np.size(x0)
	it = 0				#numero itrazioni
	x = np.copy(x0)
	err_r = np.zeros((maxit + 1, 1))	#errore relativo
	err_it = np.zeros((maxit + 1, 1))	#errore iterazioni
	err_r[0] = matrix.errRel(xTrue, x0)
	err_it[0] = 1 + tol				#inizializzato in modo che la condizione del while sia vera per almeno la prima itrazione (dove non si ha x(k-1))
	while (it < maxit and err_it[it] > tol):
		x_old = np.copy(x)
		for i in range(0, n):
	   		#x[i]=(b[i]-sum([A[i,j]*x_old[j] for j in range(0,i)])-sum([A[i, j]*x_old[j] for j in range(i+1,n)]))/A[i,i]
			x[i] = (b[i] - np.dot(A[i,0:i],x_old[0:i]) - np.dot(A[i,i+1:n],x_old[i+1:n])) / A[i,i]
		it = it + 1
		err_r[it] = matrix.errRel(xTrue, x)
		err_it[it] = matrix.errRel(x, x_old)
	err_r = err_r[:it + 1]
	err_it = err_it[1:it+1]  
	return [x, it, err_r, err_it]


def gaussSeidel(A, b, x0, maxit, tol, xTrue):
	n = np.size(x0)
	it = 0				#numero iterazioni
	x = np.copy(x0)
	err_r = np.zeros((maxit + 1, 1))	#errore relativo
	err_it = np.zeros((maxit + 1, 1))	#errore iterazioni
	err_r[0] = matrix.errRel(xTrue, x0)
	err_it[0] = 1 + tol				#inizializzato in modo che la condizione del while sia vera per almeno la prima itrazione (dove non si ha x(k-1))
	while (it < maxit and err_it[it] > tol):
		x_old = np.copy(x)
		for i in range(0, n):
			#x[i] = (b[i] - sum([A[i,j]*x[j] for j in range(0,i)]) - sum([A[i, j]*x_old[j] for j in range(i+1,n)])) / A[i,i]
			x[i] = (b[i] - np.dot(A[i,0:i], x[0:i]) - np.dot(A[i,i+1:n], x_old[i+1:n])) / A[i,i]
		it = it + 1
		err_r[it] = matrix.errRel(xTrue, x)
		err_it[it] = matrix.errRel(x, x_old)
	err_r = err_r[:it+1]
	err_it = err_it[1:it+1]  
	return [x, it, err_r, err_it]

