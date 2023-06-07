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
from scipy.optimize import minimize



from lab50_camera import *



def naive():

	FIG_SHAPE = [2,2,1]
	fig = plt.figure(figsize=Constants.FIGSIZE)

	f	= lambda _A,b,x		: matrix.errAbs( A(x.reshape(X.shape), _A), b )**2 / 2
	df	= lambda _A,b,x		: (AT( A(x.reshape(X.shape), _A), _A) - AT(b, _A)).flatten()

	f_X		= lambda x		: f(K, blurred_and_noised, x)
	df_X	= lambda x		: df(K, blurred_and_noised, x)


	x0 = np.zeros(X.size)
	max_it = 5

	res_naive = minimize(fun=f_X, x0=blurred_and_noised.flatten(), method='CG', jac=df_X, options={'maxiter':max_it,'return_all':True})
	res_naive = np.reshape(res_naive.x, X.shape)
	res_PSNR = metrics.peak_signal_noise_ratio(X, res_naive)
	res_MSE = metrics.mean_squared_error(X, res_naive)

	print(f"X: {X.shape}")
	print(f'psnr = {res_PSNR},\tmse = {res_MSE}')

	prints.img(X, FIG_SHAPE, title="Original")
	prints.img(blurred, FIG_SHAPE, title="blurred")
	prints.img(blurred_and_noised, FIG_SHAPE, title="corrupted")
	prints.img(res_naive, FIG_SHAPE, title="naive")





if __name__ == '__main__':
	naive()

	plt.show()