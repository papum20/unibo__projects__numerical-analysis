import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from numan import (
	iteration as it,
	matrix as mat,
	polynom as pol
)




##########	3	##########


functions =	(
			lambda x: x * np.exp(x),
			lambda x: 1 / (1 + 25*x),
			lambda x: np.sin(5*x) + 3*x
			)
domains =	(
			(-1, 1),
			(-1, 1),
			(1, 5)
			)
n = [1,2,3,5,7]		#gradi polinomio approssimante
N = 10				#numero punti noti
m = 300				#numero punti approssimati
x = [np.linspace(D[0],D[1],N) for D in domains]
x_plot = [np.linspace(D[0],D[1],m) for D in domains]
plot_size = (8,4)
n_err = [1,5,7]		#gradi per cui calcolare errore

# DRAW
for (f,D,xi,xi_plot) in it.rearrange_lists([functions, domains, x, x_plot]):
	plt.figure(figsize=plot_size)
	yi = pol.evaluate_fun(f, xi)
	plt.plot(xi, yi, label='real', marker='o')

	#print('xi ', xi)
	#print('yi ', yi)
	#print('xi_plot ', xi_plot)

	err0 = []
	err = []
	for deg in n:
		A = mat.vandermonde(xi, deg)
		(alphas, approx) = pol.approx(xi, yi, deg=deg)

		#print('alpha ', alphas)
		#print('approx ', approx)
		err0.append(scipy.linalg.norm(yi[0] - pol.evaluate(alphas,0)))
		if(deg in n_err): err.append(scipy.linalg.norm(np.array(yi,dtype=np.float64) - np.array(pol.evaluate_multi(alphas,xi),dtype=np.float64), 2))

		yi_plot = pol.evaluate_multi(alphas, xi_plot)
		plt.plot(xi_plot, yi_plot, label=('deg='+str(deg)) )

	print("errori in x=0: ")
	for i in range(len(err0)): print("{deg}: {err}".format(deg=n[i], err=err0[i]))
	print("errori: ")
	for i in range(len(err)): print("{deg}: {err}".format(deg=n_err[i], err=err[i]))
	
	plt.legend(fontsize=7, ncols=len(n)+1, loc='upper left')
	plt.show()
