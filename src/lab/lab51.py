import sys
sys.path.append("lib")
sys.path.append("src/lab")

from numan import (
	Constants,
	prints
)

import numpy as np
import matplotlib.pyplot as plt



from lab50_camera import *

np.random.seed(0)


FIG_SHAPE = [2,2,1]
fig = plt.figure(figsize=Constants.FIGSIZE)


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

	#psnr, mse
	print(f'psnr = {psnr},\tmse = {mse}')

	plt.show()