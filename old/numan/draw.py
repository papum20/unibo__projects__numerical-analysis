import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from . import (
	matrix as mat,
	polynom as pol
)



class RegressionPlot:
	def __init__(self, x, y, n, plot_size=(8,4), steps=300):
		self.x = x
		self.y = y
		self.n = n
		self.steps = 300
		self.plot_size = plot_size
		self.labels = ("normal equations", "SVD")
	def calculate(self):
		x, y, n = self.x, self.y, self.n
		A = mat.vandermonde(x, n)
		alphas =	[
					pol.regression_cholesky(A, y),
					pol.regression_svd(A, y)
					]
		approx = np.array([pol.evaluate_multi(a, x) for a in alphas])	# approx = (y calcolati con cholesky, y con svd)
		err = [np.linalg.norm(y - approx_i, 2) for approx_i in approx]
		self.A, self.alphas, self.approx, self.err = A, alphas, approx, err
	def debug(self):
		x, y, A, labels, alphas, err = self.x, self.y, self.A, self.labels, self.alphas, self.err
		print('Shape of x:', x.shape)
		print('Shape of y:', y.shape, "\n")
		print("A = \n", A)
		for i in range(len(labels)):
			print("alpha {} = ".format(labels[i]), alphas[i])
		(U, s, VT) = scipy.linalg.svd(A)
		print('Shape of U:', U.shape)
		print('Shape of s:', s.shape)
		print('Shape of V:', VT.T.shape)
		for i in range(len(labels)):
			print("approximation error - {} = ".format(labels[i]), err[i])
	def draw(self):
		x, y, alphas, steps, plot_size, labels = self.x, self.y, self.alphas, self.steps, self.plot_size, self.labels
		x_plot = np.linspace(x[0], x[x.size-1], steps)
		y_plots = np.array([pol.evaluate_multi(a, x_plot) for a in alphas])
		plt.figure(figsize=plot_size)
		for i in range(len(labels)):
			plt.subplot(1, 2, i+1)
			plt.title("approximation by " + labels[i])
			plt.plot(x, y)
			plt.plot(x_plot, y_plots[i])
		plt.show()
		