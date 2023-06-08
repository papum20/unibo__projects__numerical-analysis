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


cholesky_tridiagonal	= PlotStruct("cholesky-tridiagonal", (10, 1000, 20))	#step 100



for xi in cholesky_tridiagonal.axis_x:
	A = matrix.tridiagonal(xi, 9, -4, -4)
	x = np.ones((xi, 1))
	b = A@x
	cholesky_tridiagonal.K.append(np.linalg.cond(A, p=2))
	L = scipy.linalg.cholesky(A)
	y = scipy.linalg.solve(L, b)
	my_x = scipy.linalg.solve(L.T, y)
	cholesky_tridiagonal.err.append(np.linalg.norm(x - my_x, ord=2) / np.linalg.norm(x, ord=2))

printPlots([cholesky_tridiagonal])
