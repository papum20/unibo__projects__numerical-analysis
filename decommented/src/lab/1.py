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


for xi in lu_random.axis_x:
	A = np.random.randn(xi, xi)
	x = np.ones((xi, 1))
	b = A@x
	lu_random.K.append(np.linalg.cond(A, p=2))
	lu, piv = scipy.linalg.lu_factor(A)
	my_x = scipy.linalg.lu_solve((lu, piv), b)
	lu_random.err.append(np.linalg.norm(x - my_x, ord=2) / np.linalg.norm(x, ord=2))



printPlots([lu_random])
