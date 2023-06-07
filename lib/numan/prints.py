import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits import mplot3d
import numpy as np
import scipy
import scipy.linalg.decomp_lu as LUdec
from typing import Callable, Iterable
from numan import (
	Constants,
	iter,
	matrix,
	methods,
	poly
)




"""
PRINT DATAS AND PLOTS ABOUT APPROXIMATION
"""

#cholesky and svd
def approx(
	x:np.ndarray,
	y,
	n:int,	# approximation polynom degree
	steps=Constants.STEPS
):
	#N = x.size # Numero dei dati
	A = matrix.vandermonde(x, n)

	# ATA alpha=ATy
	alpha = {
		"normali"	: methods.lu(A.T@A, A.T@y),
		"svd"		: methods.svd(A, y)
	}
	my_y	= [[poly.evaluate(a, x1) for x1 in x] for a in alpha.values()]
	err_abs	= [np.absolute(np.subtract(y, my_y1)) for my_y1 in my_y]

	x_plot = np.linspace(x[0], x[-1], steps)
	y_plot = [[poly.evaluate(a, x1) for x1 in x_plot] for a in alpha.values()]

	print("A = \n", A)
	print("shape of A: ", A.shape)
	print('Shape of x:', x.shape)
	print('Shape of y:', y.shape, "\n")
	for (_key, _val) in alpha.items():
		print(f"alpha {_key} = {_val}")

	print("x: ", x)
	print("y, my_y: ")
	print(y)
	print(my_y[0])
	print(my_y[1])

	print ('Errore di approssimazione con Eq. Normali: ', err_abs[0])
	print ('Errore di approssimazione con SVD: ', err_abs[1])

	plt.rc("font", size=Constants.FONTSIZE)
	plt.figure(figsize=Constants.FIGSIZE)
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
	plt.plot(x, err_abs[0], label="normali", color="red", marker='_')
	plt.plot(x, err_abs[1], label="svd", color="blue", marker='.', linewidth=1)
	plt.xlabel("x")
	plt.ylabel("err. abs.")
	plt.legend(loc="upper left")
	plt.show()

	
#cholesky and svd
def approxMulti(
	x,
	y,
	n,	# approximation polynom degree
	steps=Constants.STEPS
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
	for (_key, _val) in alpha.items():
		print(f"alpha {_key} = {_val}")

	print ('Errore di approssimazione con Eq. Normali: ', err_a[0])
	print ('Errore di approssimazione con SVD: ', err_a[1])

	plt.rc("font", size=Constants.FONTSIZE)
	plt.figure(figsize=Constants.FIGSIZE)
	
	plt.subplot(2,2,1)
	plt.title('Approssimazione tramite Eq. Normali / SVD')
	for (y0, y1, ni) in zip(y_plot[0], y_plot[1], n):
		plt.plot(x_plot, y0, label="normali "+str(ni))
		plt.plot(x_plot, y1, label="svd "+str(ni))
	plt.plot(x, y, label="xTrue", color="green", marker="o")
	plt.legend(loc="upper left")
	plt.subplot(2,2,2)
	plt.title('Approssimazione tramite Eq. Normali')
	for y_plot_i, ni in zip(y_plot[0], n):
		plt.plot(x_plot, y_plot_i, label=str(ni))
	plt.legend(loc="upper left")
	plt.plot(x, y, label="xTrue", color="green", marker="o")
	plt.subplot(2,2,3)
	plt.title('Approssimazione tramite SVD')
	for y_plot_i, ni in zip(y_plot[1], n):
		plt.plot(x_plot, y_plot_i, label=str(ni))
	plt.legend(loc="upper left")
	plt.plot(x, y, label="xTrue", color="green", marker="o")
	plt.subplot(2,2,4)
	plt.title('errori ass per eq. normali / svd')
	for err_a0_i, err_a1_i, ni in zip(err_a[0], err_a[1], n):
		plt.plot(x, err_a0_i, label="normali"+str(ni))
		plt.plot(x, err_a1_i, label="svd"+str(ni))
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
	for i, (method, r) in enumerate(zip(methods, res)):
		print(f"{method}:")
		print("a")
		if isinstance(r[0], str): print("\tError")
		else:
			print("\txTrue: ", xTrue, "\tx found: ", r[0], "\tdiff= ", xTrue - r[0])
			print("\tf(xTrue): ", f(xTrue), "\tf(x found): ", f(r[0]), "\tdiff= ", matrix.errAbsf(f(xTrue), f(r[0])))

		if isinstance(r[2], int): print("\titerations, max_iterations: ", r[1], r[2])
		else: print("\titerations", r[1])

		if len(times) > 0:
			print("\ttime: ", times[i])

	axis_x = np.linspace(a, b, xvalues)
	axis_y = np.array([f(x) for x in axis_x])

	print("\n\n FUNCTION / SOLUTIONS \n")
	plt.rc("font", size=Constants.FONTSIZE)
	plt.figure(figsize=Constants.FIGSIZE)
	ind = 1
	plt.subplot(shape[0], shape[1], ind)
	plt.plot(axis_x, axis_y, label="f(x)")
	plt.plot(xTrue, f(xTrue), label="xTrue", marker="o")
	for xs, method in zip(res, methods):
		x = xs[0]
		if not isinstance(x, str):
			plt.subplot(shape[0], shape[1], ind)
			plt.plot(x, f(float(x)), label=method, marker="|", markersize=4)
	
	plt.title("solutions")
	plt.legend()

	print("\n\n ABSOLUTE ERROR \n")
	ind += 1
	plt.subplot(shape[0], shape[1], ind)
	for xs, method in zip(res, methods):
		if not isinstance(xs[0], str):
			axis_it = np.arange(0, xs[1], 1)
			plt.subplot(shape[0], shape[1], ind)
			plt.plot(axis_it, xs[3], label=method, marker="x")
	
	plt.title("errors (absolute) / iterations")
	plt.legend()
	plt.xlabel("iterations")
	plt.ylabel("errors (abs)")

	print("\n\ RELATIVE ERROR \n")
	ind += 1
	plt.subplot(shape[0], shape[1], ind)
	for xs, method in zip(res, methods):
		if not isinstance(xs[0], str) and isinstance(xs[2], np.ndarray):
			axis_it = np.arange(0, xs[1], 1)
			plt.subplot(shape[0], shape[1], ind)
			plt.plot(axis_it, xs[2], label=method, marker="x")
	
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
	ords:list[str]=Constants.ORDS,
	more_mat:list[np.ndarray|np.floating]=[],
	more_name:list[str]=[]
):
	A_norms = matrix.getNorms(A, ords)
	A_conds = matrix.getConds(A, ords)

	# PRINT
	print ('Norme, numeri di condizione di A:')
	for (ord,norm) in zip(ords, A_norms):
		print(f"Norma {ord} = {norm}")

	for (ord,cond) in zip(ords, A_conds):
		print(f"K(A) {ord} = {cond}")

	print("\n")

	for (mat,name) in zip(more_mat, more_name):
		print(f"{name} = {mat}")

def matEq_lu(
	A:np.ndarray,
	x:np.ndarray,
	b:np.ndarray,
	ords:list[str]=Constants.ORDS,
	more_mat:list[np.ndarray|np.floating]=[],
	more_name:list[str]=[]
):
	lu, piv = LUdec.lu_factor(A)
	# risoluzione di    Ax = b   <--->  PLUx = b 
	my_x = scipy.linalg.lu_solve((lu, piv), b)
	x_err_a = np.linalg.norm(my_x - x, ord=2)
	more_mat.extend((lu, piv, my_x, x_err_a))
	more_name.extend(("lu", "piv", "my x", "errore assoluto x"))
	matEq(A, ords, more_mat, more_name)

# richiede matrice simmetrica e definitia positiva
# A@A.T lo è sempre
def matEq_cholesky(
	A:np.ndarray,
	x:np.ndarray,
	b:np.ndarray,
	ords:list[str]=Constants.ORDS,
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
	matEq(A, ords, more_mat, more_name)


"""
OPTIMIZATION
"""

def optim(
	f:Callable[[np.ndarray], float],
	xt:list,
	yt:list,
	x_method:np.ndarray,
	labels:list[tuple[str, str]],
	shape:tuple[int, int] = (2,4)
) :
	#x = np.linspace(1,2.5,100)
	#y = np.linspace(0,1.5, 100)
	x = np.linspace(-10, 10, 500)
	y = np.linspace(-10, 10, 500)
	X, Y = np.meshgrid(x, y)
	Z = f(np.array([X, Y]))


	plt.rc("font", size=Constants.FONTSIZE)
	fig = plt.figure(figsize=Constants.FIGSIZE)
	ind = 1

	'''plots'''

	for (x1, y1, labels1) in zip(xt, yt, labels):
		#fig.add_subplot(shape[0], shape[1], ind)
		plt.subplot(shape[0], shape[1], ind)
		plt.plot(x1, y1)
		plt.xlabel(labels1[0])
		plt.ylabel(labels1[1])
		plt.title(labels1[1] + ' / ' + labels1[0])
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
	plt.plot(x_method[:,0], x_method[:,1], '*-')
	plt.title('Contour plot')

	ind += 1
	fig.add_subplot(shape[0], shape[1], ind)
	contours = plt.contour(X, Y, Z, levels=30)
	plt.plot(x_method[:,0], x_method[:,1], '*-')
	plt.title('Contour plot (flat)')
	plt.show()



""" GENERAL """

def _plot_figure(
	x:list[np.ndarray],
	y:list[np.ndarray],
	labels:list[tuple[str, str]],
	shape:tuple[int,int]=(3,4)
):
	"""
	Add plots to current figure, with x,y,labels.  
	
	## Parameters:  
	*	labels : list of lists (each for one figure) of optional tuples, in one of the following forms:  
		*	`()` : empty  
		*	`(TITLE)` :  
		*	`(XLABEL, YLABEL)` :  
		*	`(TITLE, XLABEL, YLABEL)` :  
	"""
	ind = 1
	for (x1, y1, labels1) in iter.indexSplit([x, y, labels]):
		xlabel, ylabel, title = "", "", ""
		if len(labels1) == 1:
			title = labels1[0]
		elif len(labels1) == 2:
			xlabel = labels1[0]
			ylabel = labels1[1]
			title = f'{ylabel} / {xlabel}'
		elif len(labels1) == 3:
			title = labels1[0]
			xlabel = labels1[1]
			ylabel = labels1[2]

		plt.subplot(shape[0], shape[1], ind)
		plt.plot(x1, y1)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		ind += 1
		plt.subplot(shape[0], shape[1], ind)
		plt.loglog(x1, y1)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		ind += 1
	

def plot(
	x:list[np.ndarray],
	y:list[np.ndarray],
	labels:list[tuple[str, str]],
	shape:tuple[int,int]=(3,4)
):
	"""
	Plot and show.  
	"""
	plt.rc("font", size=Constants.FONTSIZE)
	plt.figure(figsize=Constants.FIGSIZE)

	_plot_figure(x, y, labels, shape)
	plt.show()


def plot_async(
	x:list[list[np.ndarray]],
	y:list[list[np.ndarray]],
	labels:list[list[tuple[str, str]]],
	shape:tuple[int,int]=(3,4)
)->list:
	plt.rc("font", size=Constants.FONTSIZE)
	figures = []

	for (x_fg, y_fg, labels_fg) in zip(x, y, labels):
		figures.append(plt.figure(figsize=Constants.FIGSIZE))
		_plot_figure(x_fg, y_fg, labels_fg, shape)

	return figures


def img(
	img,
	shape:list[int],
	title:str=""
) -> None :
	"""
	add img to current open pyplot, increase shape index.

	Parameters
	---
	`img` : iamge
	`shape` : [nrows, ncols, index]

	-	Must be initialized and used correctly.
		Img is drawn at index, then index is increased
	"""
	ax = plt.subplot(shape[0], shape[1], shape[2])
	ax.imshow(img, cmap='gray')
	#ax.set_title(title, fontdict={'fontsize':Constants.FONTSIZE})
	plt.title(title, fontsize=Constants.FONTSIZE)
	shape[2] += 1