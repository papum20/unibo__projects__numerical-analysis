import sys
sys.path.append("lib")
sys.path.append("src/lab")

from numan import (
	Constants,
	matrix,
	polyrn,
	prints
)

import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import scipy
from skimage import data, metrics



from lab50_camera import *



def naive():

	f	= lambda _A,b,x		: matrix.errAbs( A(x.reshape(X.shape), _A), b )**2 / 2
	df	= lambda _A,b,x		: (AT( A(x.reshape(X.shape), _A), _A) - AT(b, _A)).flatten()

	f_X		= lambda x		: f(K, blurred_and_noised, x)
	df_X	= lambda x		: df(K, blurred_and_noised, x)


	max_it = 13

	res_naive, res_naive_all, res_it, res_f, res_norm, res_err_abs = polyrn.gradient(f_X, df_X, x0=blurred_and_noised.flatten(), xTrue=X.flatten(), maxit=2000)
	print(res_naive, res_naive_all)
	res_naive = np.reshape(res_naive, X.shape)

	res_PSNR = metrics.peak_signal_noise_ratio(X, res_naive)
	res_MSE = metrics.mean_squared_error(X, res_naive)

	print(f"X: {X.shape}")
	print(f'psnr = {res_PSNR},\tmse = {res_MSE}')


	FIG_SHAPE = [2,2,1]
	fig = plt.figure(figsize=Constants.FIGSIZE)

	prints.img(X, FIG_SHAPE, title="Original")
	prints.img(blurred, FIG_SHAPE, title="blurred")
	prints.img(blurred_and_noised, FIG_SHAPE, title="corrupted")
	prints.img(res_naive, FIG_SHAPE, title="naive")


	FIG_SHAPE = (2,2)

	plots_x = [ [ np.arange(0, res_it + 1) for _ in range(2) ] ]
	plots_y = [
		[
			[metrics.peak_signal_noise_ratio(X, np.reshape(xi, X.shape))	for xi in res_naive_all],
			[metrics.mean_squared_error(X, np.reshape(xi, X.shape))			for xi in res_naive_all]
		]
	]
	plots_labels = [
		[
			(
			"iterations",
			"peak signal noise ration"
			),
			(
				"iterations",
				"mean squared error"
			)
		]
	]
	plots = prints.plot_async(plots_x, plots_y, plots_labels, FIG_SHAPE)




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

				res_tikhonov, res_tikhonov_all, res_it, res_f, res_norm, res_err_abs = polyrn.gradient(f_X, df_X, x0=blurred_and_noised.flatten(), xTrue=X.flatten(), maxit=max_it)
				res_tikhonov = np.reshape(res_tikhonov, X.shape)
				prints.img(res_tikhonov, FIG_SHAPE, title="tikhonov; "+_title)

				res_PSNR = metrics.peak_signal_noise_ratio(X, res_tikhonov)
				res_MSE = metrics.mean_squared_error(X, res_tikhonov)

				#psnr, mse
				print(_title)
				print(f'psnr = {res_PSNR},\tmse = {res_MSE}')





naive()
tikhonov()
# non convergono con alcun parametro

plt.show()