import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from numan import (
	matrix as mat,
	polynom as pol
)




##########	1	##########


x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3])
y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5])
n = 5
steps = 300
plot_size = (20,10)
labels = ("eq. normali, SVD")

# CALCOLI
A = mat.vandermonde(x, n)
alphas =	[
				pol.regression_cholesky(A, y),
				pol.regression_svd(A, y)
				]
approx = np.array([pol.evaluate_multi(a, x) for a in alphas])	# approx = (y calcolati con cholesky, y con svd)
err = [np.linalg.norm(y - approx_i, 2) for approx_i in approx]

# DEBUG
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
	print("errore di approssimazione {} = ".format(labels[i]), err[i])
	
# DRAW
x_plot = np.linspace(x[0], x[x.size-1], steps)
y_plots = [np.array([[pol.evaluate_multi(a, x_plot) for a in alphas_i] for alphas_i in alphas])]
plt.figure(figsize=plot_size)
for i in range(len(labels)):
	plt.subplot(1, 2, i+1)
	plt.title("approssimazione con " + labels[i])
	plt.plot(x, y)
	plt.plot(x_plot, y_plots[i])
plt.show()
