import sys
sys.path.append("lib")
sys.path.append("../lib")

import numpy as np
import scipy
import scipy.linalg
import scipy.linalg
import matplotlib.pyplot as plt

from numan import (
	iter,
	matrix,
	prints
)


""" METODI DIRETTI """

"""1. matrici e norme """
m, n = 2, 2
A = np.array([[1, 2], [0.499, 1.001]])
x = np.ones((n, 1))
b = np.dot(A,x)

xtilde = np.array([[2.], [0.5]])
btilde = np.array([[3], [1.4985]])
deltax = np.linalg.norm(x - xtilde, ord=2)
deltab = np.linalg.norm(b - btilde, ord=2)

# Verificare che xtilde è soluzione di A xtilde = btilde
prints.matEq(A, x, b,
	more_mat=[xtilde, btilde, deltax, deltab],
	more_name=["xtilde", "btilde", "delta x", "delta b"]
)
### aggiunta
A_conds = matrix.getConds(A, prints.ORDS)
x_err_r = matrix.getRelErrs(x, xtilde - x, prints.ORDS)
b_err_r = matrix.getRelErrs(b, btilde - b, prints.ORDS)
print("err(x) <= K(A)[err(A) + err(b)] :")
print("err(x) <= K(A)err(b) : ")
for (ord,errx,K,errb) in iter.indexSplit([prints.ORDS,x_err_r,A_conds,b_err_r]):
	print("norm {ord_name}: {errx} <= {K_errb}".format(ord_name=ord, errx=errx, K_errb=K*errb))


"""2. fattorizzazione lu"""
m, n = 4, 4
A = np.array ([
	[3,-1, 1,-2],
	[0, 2, 5, -1],
	[1, 0, -7, 1],
	[0, 2, 1, 1]
])
x = np.ones(n)
b = A@x

prints.matEq_lu(A, x, b)


"""3. Choleski """
# richiede matrice simmetrica e definitia positiva
# A@A.T lo è sempre
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



"""4. Choleski con matrice di Hilbert"""
# la matrice di Hilbert è simmetrica e definita positiva

n = np.random.randint(2, 14, size=None, dtype=int)
A = scipy.linalg.hilbert(n)
x = np.ones((n, 1))
b = A@x

prints.matEq_cholesky(A, x, b)


"""5. Choleski con matrice tridiagonale simmetrica e definita positiva """
n = np.random.randint(2, 16)
A = matrix.tridiagonal(n, 9, -4, -4)
A = np.matmul(A, np.transpose(A))
x = np.ones((n,1))
b = A@x

prints.matEq_cholesky(A, x, b)


"""6. plots """
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
		
def printPlot(S:PlotStruct):
	axis_x, K, err = S.axis_x, S.K, S.err
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

def printPlots(S:list[PlotStruct]):
	plt.figure(figsize=FIGSIZE)
	for i in range(len(S)):
		plt.subplot(2, 2, i+1)
		plt.semilogy(S[i].axis_x, S[i].K)
		plt.title("condizionamento / dimensione | " + S[i].name, fontsize=FONTSIZE)
		plt.xlabel("dimensione matrice", fontsize=FONTSIZE)
		plt.ylabel("condizionamento", fontsize=FONTSIZE)
	plt.show()
	plt.figure(figsize=FIGSIZE)
	for i in range(len(S)):
		plt.subplot(2, 2, i+1)
		plt.plot(S[i].axis_x, S[i].err)
		plt.title("errore relativo / dimensione | " + S[i].name, fontsize=FONTSIZE)
		plt.xlabel("dimensione matrice", fontsize=FONTSIZE)
		plt.ylabel("errore relativo", fontsize=FONTSIZE)
	plt.show()

# lu, random
R = PlotStruct("lu-random", (10, 1000, 100))
# cholesky, hilbert
H = PlotStruct("cholesky-hilbert", (2, 13, 1))
#cholesky, tridiagonal
D = PlotStruct("cholesky-tridiagonal", (10, 1000, 100))

## LU
for n in R.axis_x:
	A = np.random.randn(n, n)
	x = np.ones((n, 1))
	b = A@x
	R.K.append(np.linalg.cond(A, p=2))
	lu, piv = scipy.linalg.lu_factor(A)
	my_x = scipy.linalg.lu_solve((lu, piv), b)
	R.err.append(np.linalg.norm(x - my_x, ord=2) / np.linalg.norm(x, ord=2))

## HILBERT
"""
LA MATRICE DI HILBERT È DEFINITA POSITIVA, MA È MOLTO MAL CONDIZIONATA,
PER CUI CON L'AUMENTARE DELLE DIMENSIONI (>=15) PER ERRORI ALGORITMICI/DI ARROTONDAMENTPO
RISULTA NON DEFINITA POSITIVA
"""
for n in H.axis_x:
	A = scipy.linalg.hilbert(n)
	x = np.ones((n, 1))
	b = A@x
	H.K.append(np.linalg.cond(A, p=2))
	L = scipy.linalg.cholesky(A)
	y = scipy.linalg.solve(L, b)
	my_x = scipy.linalg.solve(L.T, y)
	H.err.append(np.linalg.norm(x - my_x, ord=2) / np.linalg.norm(x, ord=2))


## TRIDIAGONALE
for n in D.axis_x:
	A = matrix.tridiagonal(n, 9, -4, -4)
	x = np.ones((n, 1))
	b = A@x
	D.K.append(np.linalg.cond(A, p=2))
	L = scipy.linalg.cholesky(A)
	y = scipy.linalg.solve(L, b)
	my_x = scipy.linalg.solve(L.T, y)
	D.err.append(np.linalg.norm(x - my_x, ord=2) / np.linalg.norm(x, ord=2))

printPlots([R, H, D])
