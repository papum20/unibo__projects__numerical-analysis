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



from lab50_camera import *

np.random.seed(0)


FIG_SHAPE = [2,2,1]
fig = plt.figure(figsize=Constants.FIGSIZE)

f	= lambda A,b,x		: matrix.errAbs(A@x, b)**2 / 2
df	= lambda A,AT,b,x	: AT@A@x - AT@b
f_camera	= lambda x: f(K, noised, x)
df_camera	= lambda x: df(K, K.T, noised, x)

x0 = np.zeros(camera.size)
naive = minimize(f_camera, x0, jac=df_camera)

if __name__ == '__main__':
	#camera
	print(f"camera: {camera.shape}")
	prints.img(camera, FIG_SHAPE)
	#kernel
	print(f"kernel: {ker.shape}")
	print(ker)
	#K
	print(f"K: {K.shape}")
	print(K)
	# blur
	print(f"blurred: {blurred.shape}")
	print(blurred)
	prints.img(blurred, FIG_SHAPE)

	# noise
	print(f"noise: {noise.shape}")
	print(noise)
	print(f"noised: {noised.shape}")
	print(noised)
	prints.img(noised, FIG_SHAPE)

	# naive
	print(f"naive: {naive.shape}")
	print(naive)
	prints.img(naive, FIG_SHAPE)

	#psnr, mse
	print(f'psnr = {psnr},\tmse = {mse}')

	plt.show()