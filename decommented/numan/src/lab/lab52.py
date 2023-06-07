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
from scipy.optimize import minimize
from skimage import data, metrics




from lab50_camera import *



def naive():

	f	= lambda _A,b,x		: matrix.errAbs( A(x.reshape(X.shape), _A), b )**2 / 2
	df	= lambda _A,b,x		: (AT( A(x.reshape(X.shape), _A), _A) - AT(b, _A)).flatten()

	f_X		= lambda x		: f(K, blurred_and_noised, x)
	df_X	= lambda x		: df(K, blurred_and_noised, x)


	max_it = 13

	res_naive_dict = minimize(fun=f_X, x0=blurred_and_noised.flatten(), method='CG', jac=df_X, options={'maxiter':max_it,'return_all':True})
	res_naive = np.reshape(res_naive_dict.x, X.shape)
	res_naive_all = res_naive_dict.allvecs

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

	plots_x = [ [ np.arange(0, res_naive_dict.nit + 1) for _ in range(2) ] ]
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




if __name__ == '__main__':
	naive()

	plt.show()