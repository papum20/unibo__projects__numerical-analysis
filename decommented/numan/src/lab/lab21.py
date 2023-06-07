import sys
sys.path.append("lib")

import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

from numan import (
	Constants,
	matrix,
	prints
)


""" METODI DIRETTI """

print("""\n\n1. matrici e norme \n""")
xi = 2
A = np.array([[1, 2], [0.499, 1.001]])
x = np.ones((xi, 1))
b = np.dot(A,x)

xtilde = np.array([[2.], [0.5]])
btilde = np.array([[3], [1.4985]])
deltax = np.linalg.norm(x - xtilde, ord=2)
deltab = np.linalg.norm(b - btilde, ord=2)

prints.matEq(A,
	more_mat=[xtilde, btilde, deltax, deltab],
	more_name=["xtilde", "btilde", "delta x", "delta b"]
)

A_conds = matrix.getConds(A, Constants.ORDS)
x_err_r = matrix.getRelErrs(x, xtilde - x, Constants.ORDS)
b_err_r = matrix.getRelErrs(b, btilde - b, Constants.ORDS)
print("err(x) <= K(A)[err(A) + err(b)] :")
print("err(x) <= K(A)err(b) : ")
for (ord, errx, K, errb) in zip(Constants.ORDS, x_err_r, A_conds, b_err_r):
	print("norm {ord_name}: {errx} <= {K_errb}".format(ord_name=ord, errx=errx, K_errb=K*errb))

print (f'Verifica: A*xtilde = {A@xtilde} = {btilde} = btilde')


print("""\n\n2. fattorizzazione lu\n""")
m, xi = 4, 4
A = np.array ([
	[3,-1, 1,-2],
	[0, 2, 5, -1],
	[1, 0, -7, 1],
	[0, 2, 1, 1]
])
x = np.ones(xi)
b = A@x

prints.matEq_lu(A, x, b)


print("""\n\n3. Choleski \n""")


A = np.array ([
	[3,-1, 1,-2],
	[0, 2, 5, -1],
	[1, 0, -7, 1],
	[0, 2, 1, 1] 
	],
	dtype=np.float64
)
A = np.matmul(A, np.transpose(A))
x = np.ones((4,1))
b = A@x

prints.matEq_cholesky(A, x, b)



print("""\n\n4. Choleski con matrice di Hilbert\n""")

xi = np.random.randint(2, 14, size=None, dtype=int)
A = scipy.linalg.hilbert(xi)
x = np.ones((xi, 1))
b = A@x

prints.matEq_cholesky(A, x, b)


print("""\n\n5. Choleski con matrice tridiagonale simmetrica e definita positiva \n""")
xi = np.random.randint(2, 16)
A = matrix.tridiagonal(xi, 9, -4, -4)
A = np.matmul(A, np.transpose(A))
x = np.ones((xi,1))
b = A@x

prints.matEq_cholesky(A, x, b)


print("""\n\n6. plots \n""")
FIGSIZE = (15,7)
DPI = 100
FONTSIZE = 8

class PlotStruct:
	name = ""
	size = ()
	axis_x = []
	K = []
	err = []
	def __init__(self, name:str, size):
		self.name = name
		self.size = size
		self.axis_x = np.arange(size[0], size[1], size[2])	
		self.K = []
		self.err = []
		
def printPlot(struct:PlotStruct):
	axis_x, K, err = struct.axis_x, struct.K, struct.err
	plt.semilogy(axis_x, K)
	plt.title("condizionamento / dimensione")
	plt.xlabel("dimensione matrice")
	plt.ylabel("condizionamento")
	plt.show()
	plt.plot(axis_x, err)
	plt.title("errore relativo / dimensione")
	plt.xlabel("dimensione matrice")
	plt.ylabel("errore relativo")
	plt.show()

def printPlots(structs:list[PlotStruct]):
	plt.figure(figsize=FIGSIZE)
	for i, struct in enumerate(structs):
		plt.subplot(2, 2, i+1)
		plt.semilogy(struct.axis_x, struct.K)
		plt.title("condizionamento / dimensione | " + struct.name, fontsize=FONTSIZE)
		plt.xlabel("dimensione matrice", fontsize=FONTSIZE)
		plt.ylabel("condizionamento", fontsize=FONTSIZE)
	plt.figure(figsize=FIGSIZE)
	for i, struct in enumerate(structs):
		plt.subplot(2, 2, i+1)
		plt.plot(struct.axis_x, struct.err)
		plt.title("errore relativo / dimensione | " + struct.name, fontsize=FONTSIZE)
		plt.xlabel("dimensione matrice", fontsize=FONTSIZE)
		plt.ylabel("errore relativo", fontsize=FONTSIZE)
	plt.show()


lu_random				= PlotStruct("lu-random", (10, 1000, 20))	#step 100
cholesky_hilbert		= PlotStruct("cholesky-hilbert", (2, 13, 1))
cholesky_tridiagonal	= PlotStruct("cholesky-tridiagonal", (10, 1000, 20))	#step 100

for xi in lu_random.axis_x:
	A = np.random.randn(xi, xi)
	x = np.ones((xi, 1))
	b = A@x
	lu_random.K.append(np.linalg.cond(A, p=2))
	lu, piv = scipy.linalg.lu_factor(A)
	my_x = scipy.linalg.lu_solve((lu, piv), b)
	lu_random.err.append(np.linalg.norm(x - my_x, ord=2) / np.linalg.norm(x, ord=2))



for xi in cholesky_hilbert.axis_x:
	A = scipy.linalg.hilbert(xi)
	x = np.ones((xi, 1))
	b = A@x
	cholesky_hilbert.K.append(np.linalg.cond(A, p=2))
	L = scipy.linalg.cholesky(A)
	y = scipy.linalg.solve(L, b)
	my_x = scipy.linalg.solve(L.T, y)
	cholesky_hilbert.err.append(np.linalg.norm(x - my_x, ord=2) / np.linalg.norm(x, ord=2))


for xi in cholesky_tridiagonal.axis_x:
	A = matrix.tridiagonal(xi, 9, -4, -4)
	x = np.ones((xi, 1))
	b = A@x
	cholesky_tridiagonal.K.append(np.linalg.cond(A, p=2))
	L = scipy.linalg.cholesky(A)
	y = scipy.linalg.solve(L, b)
	my_x = scipy.linalg.solve(L.T, y)
	cholesky_tridiagonal.err.append(np.linalg.norm(x - my_x, ord=2) / np.linalg.norm(x, ord=2))

printPlots([lu_random, cholesky_hilbert, cholesky_tridiagonal])
