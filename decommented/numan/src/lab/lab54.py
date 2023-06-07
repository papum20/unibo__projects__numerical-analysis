import sys
sys.path.append("lib")

from numan import (
	prints
)

import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from skimage import data, metrics




def gaussian_kernel(kernlen, sigma):
	x = np.linspace(- (kernlen // 2), kernlen // 2, kernlen)    
	kern1d = np.exp(- 0.5 * (x**2 / sigma))
	kern2d = np.outer(kern1d, kern1d)
	return kern2d / kern2d.sum()

def psf_fft(K, d, shape):
	K_p = np.zeros(shape)
	K_p[:d, :d] = K

	p = d // 2
	K_pr = np.roll(np.roll(K_p, -p, 0), -p, 1)

	K_otf = fft.fft2(K_pr)
	return K_otf


def A(x, K):
	x = fft.fft2(x)
	return np.real(fft.ifft2(K * x))


def AT(x, K):
	x = fft.fft2(x)
	return np.real(fft.ifft2(np.conj(K) * x))

def get_gray_image(X):
	if len(X.shape) == 3:
		m, n, k = X.shape
		res = np.zeros((m, n))

		for i in range(m):
			for j in range(n):
				res[i][j] = X[i][j][0]
		return res
	else:
		return X



from lab51 import corrupted
from lab52 import naive
from lab53 import tikhonov


#X = data.camera().astype(np.float64) / 255.0 # type: ignore
X = data.cat().astype(np.float64) / 255.0 # type: ignore
#X = data.horse().astype(np.float64) / 255.0 # type: ignore

X = get_gray_image(X)

kernlens_sigmas = (
	(5, .5),
	(7, .1),
	(9, 1.3)
)

for (kernlen, sigma) in kernlens_sigmas:

	noise_sigma = 0.03
	ker = gaussian_kernel(kernlen, sigma)
	K = psf_fft(ker, kernlen, X.shape)

	blurred = A(X, K)

	noise = np.random.normal(loc=0, scale=noise_sigma, size=X.shape)
	blurred_and_noised = blurred + noise
	blurred_and_noisedT = AT(blurred_and_noised, K)

	psnr = metrics.peak_signal_noise_ratio(X, blurred_and_noised)
	mse = metrics.mean_squared_error(X, blurred_and_noised)


	corrupted()
	naive()
	tikhonov()

plt.show()