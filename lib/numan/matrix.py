import numpy as np
import scipy


### SPECIAL MATRIXES

def tridiagonal(n, d_val, d_val_u, d_val_d):    #d=val(_u/_d): valore su diagonale (up/down)
    #A = np.diag(np.ones(n) * D_val, k=0) + np.diag(np.ones(n-1) * D-val_u, k=1) + np.diag(np.ones(n-1) * D_val_d, k=-1)    
    return np.eye(n,k=0)*d_val + np.eye(n,k=1)*d_val_u + np.eye(n,k=-1)*d_val_d

# x = [[x0..xm]]
# deg = max degree (=n)
def vandermonde(x, deg):
    return np.array([x**j for j in range(deg+1)]).T




### GET (SINGLE)

def condition(A:np.ndarray, ord=2):
	return scipy.linalg.norm(np.linalg.inv(A), ord=ord) * scipy.linalg.norm(A, ord=ord)

def errAbs(A:np.ndarray, A2:np.ndarray):
	return scipy.linalg.norm(np.subtract(A, A2), ord=2)
def errAbsf(A:float, A2:float):
	return scipy.linalg.norm([A-A2], ord=2)
def errRel(A:np.ndarray, A2:np.ndarray):
	return scipy.linalg.norm(np.subtract(A, A2), ord=2) / scipy.linalg.norm(A, ord=2)

def spectralRadius(A:np.ndarray):
	return max(list(map(lambda x: abs(x), np.linalg.eigvals(A))))


### GET (MULTIPLE)

def getConds(A, ords={"1", "2", "fro", "inf"}):
    res = []
    if("1" in ords): res.append(np.linalg.cond(A, 1))
    if("2" in ords): res.append(np.linalg.cond(A, 2))
    if("fro" in ords): res.append(np.linalg.cond(A, "fro"))
    if("inf" in ords): res.append(np.linalg.cond(A, np.inf))
    return res

def getRelErrs(A, diff, ords={"1", "2", "fro", "inf"}):
    res = []
    if("1" in ords): res.append(np.linalg.norm(diff, ord=1) / np.linalg.norm(A, ord=1))
    if("2" in ords): res.append(np.linalg.norm(diff, ord=2) / np.linalg.norm(A, ord=2))
    if("fro" in ords): res.append(np.linalg.norm(diff, ord="fro") / np.linalg.norm(A, ord="fro"))
    if("inf" in ords): res.append(np.linalg.norm(diff, ord=np.inf) / np.linalg.norm(A, ord=np.inf))
    return res

def getNorms(A, ords={"1", "2", "fro", "inf"}):
    res = []
    if("1" in ords): res.append(np.linalg.norm(A, ord=1))
    if("2" in ords): res.append(np.linalg.norm(A, ord=2))
    if("fro" in ords): res.append(np.linalg.norm(A, ord="fro"))
    if("inf" in ords): res.append(np.linalg.norm(A, ord=np.inf))
    return res

