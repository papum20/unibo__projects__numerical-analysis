import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits import mplot3d
import numpy as np
import scipy
import scipy.linalg.decomp_lu as LUdec
from typing import Callable, Iterable
from numan import (
	iter,
	matrix,
	methods,
	poly
)
import numan



"""
PRINT DATAS AND PLOTS ABOUT APPROXIMATION
"""

#cholesky and svd
def approx(
	x:np.ndarray,
	y,
	n:int,	# approximation polynom degree
	steps=numan.STEPS
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

	plt.rc("font", size=numan.FONTSIZE)
	plt.figure(figsize=numan.FIGSIZE)
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
	steps=numan.STEPS
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

	plt.rc("font", size=numan.FONTSIZE)
	plt.figure(figsize=numan.FIGSIZE)
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
SEARCH FOR FUNCTION SOLUTION
"""

def funSolve(
	a:float,
	b:float,
	f:Callable[[float], float],
	xTrue:float,
	methods:list[str],										#names, in same order as res
	res:list[ tuple[float|str, int, int, np.ndarray] | tuple[float|str, int, np.ndarray, np.ndarray] ],	#results calculated with methods
	times:list = [],
	xvalues:int = 100,
	shape:tuple[int, int] = (int(2), int(2))
):
	for (m,r,time) in iter.indexSplit([methods,res,times]):
		print("{}:".format(m))
		#IF ERROR
		if type(r[0]) == type(str): print("\tError")
		else:
			#SOLUTION
			print("\txTrue: ", xTrue, "\tx found: ", r[0], "\tdiff= ", xTrue - r[0])
			print("\tf(xTrue): ", f(xTrue), "\tf(x found): ", f(r[0]), "\tdiff= ", matrix.errAbsf(f(xTrue), f(r[0])))
		#ITERATIONS, ERRORS
		if type(r[2]) == int: print("\titerations, max_iterations: ", r[1], r[2])
		else: print("\titerations", r[1])
		#TIMES
		if len(times) > 0:
			print("\ttime: ", time)

	axis_x = np.linspace(a, b, xvalues)
	axis_y = np.array([f(x) for x in axis_x])

	## FUNCTION / SOLUTIONS
	plt.rc("font", size=numan.FONTSIZE)
	plt.figure(figsize=numan.FIGSIZE)
	ind = 1
	plt.subplot(shape[0], shape[1], ind)
	plt.plot(axis_x, axis_y, label="f(x)")
	plt.plot(xTrue, f(xTrue), label="xTrue", marker="o")
	for i in range(len(methods)):
		x = res[i][0]
		if(type(x) != str):
			plt.subplot(shape[0], shape[1], ind)
			plt.plot(x, f(float(x)), label=methods[i], marker="|", markersize=4)
	
	plt.title("solutions")
	plt.legend()
	## ABSOLUTE ERROR
	ind += 1
	plt.subplot(shape[0], shape[1], ind)
	for i in range(len(methods)):
		if(type(res[i][0]) != str):
			axis_it = np.arange(0, res[i][1], 1)
			plt.subplot(shape[0], shape[1], ind)
			plt.plot(axis_it, res[i][3], label=methods[i], marker="x")
	
	plt.title("errors (absolute) / iterations")
	plt.legend()
	plt.xlabel("iterations")
	plt.ylabel("errors (abs)")
	## RELATIVE ERROR
	ind += 1
	plt.subplot(shape[0], shape[1], ind)
	for i in range(len(methods)):
		if type(res[i][0]) != str and type(res[i][2]) == np.ndarray:
			axis_it = np.arange(0, res[i][1], 1)
			plt.subplot(shape[0], shape[1], ind)
			plt.plot(axis_it, res[i][2], label=methods[i], marker="x")
	
	plt.title("errors (between iterations) / iterations")
	plt.legend()
	plt.xlabel("iterations")
	plt.ylabel("errors (it)")

	plt.show()
	



"""
PRINT DATA ON MATRIX EQUATIONS
"""

def matEq(
	A:np.ndarray,
	x:np.ndarray,
	b:np.ndarray,
	ords:list[str]=numan.ORDS,
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
	ords:list[str]=numan.ORDS,
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
	ords:list[str]=numan.ORDS,
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


"""
OPTIMIZATION
"""

def optim(
	f:Callable[[np.ndarray], float],
	xt:list,
	yt:list,
	labels:list[tuple[str, str]],
	shape:tuple[int, int] = (2,3)
) :
	x = np.linspace(1,2.5,100)
	y = np.linspace(0,1.5, 100)
	X, Y = np.meshgrid(x, y)
	Z = f(np.array([X, Y]))


	plt.rc("font", size=numan.FONTSIZE)
	fig = plt.figure(figsize=numan.FIGSIZE)
	ind = 1

	'''plots'''

	for (x1, y1, labels1) in iter.indexSplit([xt, yt, labels]):
		#fig.add_subplot(shape[0], shape[1], ind)
		plt.subplot(shape[0], shape[1], ind)
		plt.plot(x1, y1)
		plt.xlabel(labels1[0])
		plt.ylabel(labels1[1])
		plt.title(labels1[0] + ' / ' + labels1[1])
		ind += 1

	# 3d plots
	ax1 = fig.add_subplot(shape[0], shape[1], ind, projection='3d')
	#ax1 = plt.axes(projection='3d')
	ax1.plot_surface(X, Y, Z, cmap='viridis')
	ax1.set_title('Surface plot')
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	ax1.view_init(elev=20)

	ind += 1
	ax2 = fig.add_subplot(shape[0], shape[1], ind, projection='3d')
	ax2.plot_surface(X, Y, Z, cmap='viridis')
	ax2.set_title('Surface plot from a different view')
	ax2.set_xlabel("x")
	ax2.set_ylabel("y")
	ax2.view_init(elev=5)

	# contours
	ind += 1
	fig.add_subplot(shape[0], shape[1], ind, projection="3d")
	contours = plt.contour(X, Y, Z, levels=30)
	plt.title('Contour plot')
	plt.show()



""" GENERAL """

def plot(
	x:list[np.ndarray],
	y:list[np.ndarray],
	labels:list[tuple[str, str]],
	shape:tuple[int,int]=(3,4)
):
	plt.rc("font", size=numan.FONTSIZE)
	plt.figure(figsize=numan.FIGSIZE)
	ind = 1

	for (x1, y1, labels1) in iter.indexSplit([x, y, labels]):
		plt.subplot(shape[0], shape[1], ind)
		plt.plot(x1, y1)
		plt.xlabel(labels1[0])
		plt.ylabel(labels1[1])
		plt.title(labels1[0] + ' / ' + labels1[1])
		ind += 1
		plt.subplot(shape[0], shape[1], ind)
		plt.loglog(x1, y1)
		plt.xlabel(labels1[0])
		plt.ylabel(labels1[1])
		plt.title(labels1[0] + ' / ' + labels1[1])
		ind += 1
	plt.show()