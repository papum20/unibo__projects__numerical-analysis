import sys
sys.path.append("lib")
sys.path.append("src/lab")

from numan import (
	Constants,
	matrix,
	prints
)

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize



from lab50_camera import *



def tikhonov():

	def f(_A,b,x,lam):
		x = x.reshape(X.shape)
		return matrix.errAbs( AT(x, _A), b )**2 / 2 + (lam / 2) * scipy.linalg.norm(x, ord=2)**2
		
	def df(_A,b,x, lam):
		x = x.reshape(X.shape)
		return (AT( A(x, _A), _A) - AT(b, _A) + lam * x).flatten()


	lambdas = (0.0001, 0.01, 2, 200)
	ites = (5, 13)


	#figures = []
	FIG_SHAPE = [1,6,1]
	fig = plt.figure(figsize=Constants.FIGSIZE)
	prints.img(X, FIG_SHAPE, title="Original")

	for max_it in ites:
		f_X 	= lambda x	: f(K, blurred_and_noised, x, lam)
		df_X	= lambda x	: df(K, blurred_and_noised, x, lam)

		FIG_SHAPE = [1,6,1]
		fig = plt.figure(figsize=Constants.FIGSIZE)


		for _lam_i, lam in enumerate(lambdas):

			#FIG_SHAPE = [2,2,1]
			#figures.append(plt.figure(figsize=Constants.FIGSIZE))
			

				_title = "lambda: {}, it: {}".format(lam, max_it)

				#prints.img(X, FIG_SHAPE)
				#prints.img(blurred, FIG_SHAPE)

				res_tikhonov = minimize(fun=f_X, x0=blurred_and_noised.flatten(), method='CG', jac=df_X, options={'maxiter':max_it,'return_all':True})
				res_tikhonov = np.reshape(res_tikhonov.x, X.shape)
				prints.img(res_tikhonov, FIG_SHAPE, title="tikhonov; "+_title)

				res_PSNR = metrics.peak_signal_noise_ratio(X, res_tikhonov)
				res_MSE = metrics.mean_squared_error(X, res_tikhonov)

				#psnr, mse
				print(_title)
				print(f'psnr = {res_PSNR},\tmse = {res_MSE}')



if __name__ == '__main__':
	tikhonov()
	plt.show()