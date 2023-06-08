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


def corrupted():

	FIG_SHAPE = [2,2,1]
	fig = plt.figure(figsize=Constants.FIGSIZE)

	print(f"\nX: {X.shape}")

	prints.img(X, FIG_SHAPE, title="original")
	prints.img(blurred, FIG_SHAPE, title="blurred")
	prints.img(blurred_and_noised, FIG_SHAPE, title="blurred and noised")

	print(f'psnr = {psnr},\tmse = {mse}')

	

if __name__ == '__main__':
	corrupted()
	plt.show()