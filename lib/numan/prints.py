import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg.decomp_lu as LUdec
from numan import (
	iter,
	matrix,
	methods,
	poly
)

ORDS = ["1", "2", "fro", "inf"]
FIGSIZE = (15,7)
DPI = 100
FONTSIZE = 7
STEPS = 300


"""
PRINT DATAS AND PLOTS ABOUT APPROXIMATION
"""

#cholesky and svd
def approx(
	x:np.ndarray,
	y,
	n:int,	# approximation polynom degree
	steps=STEPS
):
	N = x.size # Numero dei dati
	A = matrix.vandermonde(x, n)

	""" risoluzione 2 metodi """
	# ATA alpha=ATy
	alpha = {
		"normali" : methods.lu(A.T@A, A.T@y),
		"svd" : methods.svd(A, y)
	}
	my_y = [[poly.evaluate(a, x1) for x1 in x] for a in alpha.values()]
	err_a = [np.absolute(np.subtract(y, my_y1)) for my_y1 in my_y]

	x_plot = np.linspace(x[0], x[x.size-1], steps)
	y_plot = [[poly.evaluate(a, x1) for x1 in x_plot] for a in alpha.values()]

	print("A = \n", A)
	print("shape of A: ", A.shape)
	print('Shape of x:', x.shape)
	print('Shape of y:', y.shape, "\n")
	for a in alpha:
		print("alpha {} = {}".format(a, alpha[a]))

	print("x: ", x)
	print("y, my_y: ")
	print(y)
	print(my_y[0])
	print(my_y[1])

	print ('Errore di approssimazione con Eq. Normali: ', err_a[0])
	print ('Errore di approssimazione con SVD: ', err_a[1])

	plt.rc("font", size=FONTSIZE)
	plt.figure(figsize=FIGSIZE)
	plt.subplot(2,2,1)
	plt.title('Approssimazione tramite Eq. Normali / SVD')
	plt.plot(x_plot, y_plot[0], label="normali", color="red", marker='_')
	plt.plot(x_plot, y_plot[1], label="svd", color="blue", marker='.', linewidth=1)
	plt.plot(x, y, label="xTrue", color="green", marker="o")
	plt.legend(loc="upper left")
	plt.subplot(2,2,2)
	plt.title('Approssimazione tramite Eq. Normali')
	plt.plot(x_plot, y_plot[0], label="normali", color="red", marker='_')
	plt.plot(x, y, label="xTrue", color="green", marker="o")
	plt.subplot(2,2,3)
	plt.title('Approssimazione tramite SVD')
	plt.plot(x_plot, y_plot[1], label="svd", color="blue", marker='.', linewidth=1)
	plt.plot(x, y, label="xTrue", color="green", marker="o")
	plt.subplot(2,2,4)
	plt.title('errori ass per eq. normali / svd')
	plt.plot(x, err_a[0], label="normali", color="red", marker='_')
	plt.plot(x, err_a[1], label="svd", color="blue", marker='.', linewidth=1)
	plt.xlabel("x")
	plt.ylabel("err. abs.")
	plt.legend(loc="upper left")
	plt.show()

	
#cholesky and svd
def approxMulti(
	x,
	y,
	n,	# approximation polynom degree
	steps=STEPS
):
	N = x.size # Numero dei dati
	A = [matrix.vandermonde(x, n1) for n1 in n]

	""" risoluzione 2 metodi """
	# ATA alpha=ATy
	alpha = {
		"normali" : [methods.cholesky(A1.T@A1, A1.T@y) for A1 in A],
		"svd" : [methods.svd(A1, y) for A1 in A]
	}
	my_y = [[[poly.evaluate(a_n, x1) for x1 in x] for a_n in method] for method in alpha.values()]
	err_a = [[np.absolute(np.subtract(y, y_n)) for y_n in method] for method in my_y]

	x_plot = np.linspace(x[0], x[x.size-1], steps)
	y_plot = [[[poly.evaluate(a_n, x1) for x1 in x_plot] for a_n in a_method] for a_method in alpha.values()]

	print('Shape of x:', x.shape)
	print('Shape of y:', y.shape, "\n")
	for a in alpha:
		print("alpha {} = {}".format(a, alpha[a]))

	print ('Errore di approssimazione con Eq. Normali: ', err_a[0])
	print ('Errore di approssimazione con SVD: ', err_a[1])

	plt.rc("font", size=FONTSIZE)
	plt.figure(figsize=FIGSIZE)
	plt.subplot(2,2,1)
	plt.title('Approssimazione tramite Eq. Normali / SVD')
	for i in range(len(n)):
		plt.plot(x_plot, y_plot[0][i], label="normali "+str(n[i]))
		plt.plot(x_plot, y_plot[1][i], label="svd "+str(n[i]))
	plt.plot(x, y, label="xTrue", color="green", marker="o")
	plt.legend(loc="upper left")
	plt.subplot(2,2,2)
	plt.title('Approssimazione tramite Eq. Normali')
	for i in range(len(n)):
		plt.plot(x_plot, y_plot[0][i], label=str(n[i]))
	plt.legend(loc="upper left")
	plt.plot(x, y, label="xTrue", color="green", marker="o")
	plt.subplot(2,2,3)
	plt.title('Approssimazione tramite SVD')
	for i in range(len(n)):
		plt.plot(x_plot, y_plot[1][i], label=str(n[i]))
	plt.legend(loc="upper left")
	plt.plot(x, y, label="xTrue", color="green", marker="o")
	plt.subplot(2,2,4)
	plt.title('errori ass per eq. normali / svd')
	for i in range(len(n)):
		plt.plot(x, err_a[0][i], label="normali"+str(n[i]))
		plt.plot(x, err_a[1][i], label="svd"+str(n[i]))
	plt.xlabel("x")
	plt.ylabel("err. abs.")
	plt.legend(loc="upper left")
	plt.show()



"""
PRINT DATA ON MATRIX EQUATIONS
"""

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

