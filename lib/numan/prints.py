import numpy as np
import scipy
import scipy.linalg.decomp_lu as LUdec
from numan import (
	iter,
	matrix
)

ORDS = ["1", "2", "fro", "inf"]



def matEq(
	A:np.ndarray,
	x:np.ndarray,
	b:np.ndarray,
	ords:list[str]=ORDS,
	more_mat:list[np.ndarray|np.floating]=[],
	more_name:list[str]=[]
):
	A_norms = matrix.getNorms(A, ords)
	A_conds = matrix.getConds(A, ords)

	# PRINT
	print ('Norme, numeri di condizione di A:')
	for (ord,norm) in iter.indexSplit([ords,A_norms]):
		print("Norma {ord_name} = {norm_val}".format(ord_name=ord, norm_val=norm))

	for (ord,cond) in iter.indexSplit([ords,A_conds]):
		print("K(A) {ord_name} = {cond_val}".format(ord_name=ord, cond_val=cond))

	print("\n")

	for (mat,name) in iter.indexSplit([more_mat,more_name]):
		print("{name} = {mat}".format(name=name,mat=mat))

def matEq_lu(
	A:np.ndarray,
	x:np.ndarray,
	b:np.ndarray,
	ords:list[str]=ORDS,
	more_mat:list[np.ndarray|np.floating]=[],
	more_name:list[str]=[]
):
	lu, piv = LUdec.lu_factor(A)
	# risoluzione di    Ax = b   <--->  PLUx = b 
	my_x = scipy.linalg.lu_solve((lu, piv), b)
	x_err_a = np.linalg.norm(my_x - x, ord=2)
	more_mat.extend((lu,piv,my_x,x_err_a))
	more_name.extend(("lu","piv","my x","errore assoluto x"))
	matEq(A, x, b, ords, more_mat, more_name)

# richiede matrice simmetrica e definitia positiva
# A@A.T lo è sempre
def matEq_cholesky(
	A:np.ndarray,
	x:np.ndarray,
	b:np.ndarray,
	ords:list[str]=ORDS,
	more_mat:list[np.ndarray|np.floating]=[],
	more_name:list[str]=[]
):
	# decomposizione di Choleski
	L = scipy.linalg.cholesky(A, lower=True)
	A_chol = L@L.T
	A_err_a = scipy.linalg.norm(A - A_chol, ord='fro')
	y = scipy.linalg.solve(L.T, b)
	x_chol = scipy.linalg.solve(L, y)
	x_err_a = scipy.linalg.norm(x - x_chol, ord='fro')
	more_mat.extend((L, A_chol, A_err_a, y, x_chol, x_err_a))
	more_name.extend(("L", "A_chol", "A_err_a", "y", "x_chol", "x_err_a"))
	matEq(A, x, b, ords, more_mat, more_name)
