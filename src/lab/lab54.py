import sys
sys.path.append("lib")

from numan import (
	Constants,
	prints
)

import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from skimage import data, metrics



np.random.seed(0)

# Crea un kernel Gaussiano di dimensione kernlen e deviazione standard sigma
def gaussian_kernel(kernlen, sigma):
	x = np.linspace(- (kernlen // 2), kernlen // 2, kernlen)    
	# Kernel gaussiano unidmensionale
	kern1d = np.exp(- 0.5 * (x**2 / sigma))
	# Kernel gaussiano bidimensionale
	kern2d = np.outer(kern1d, kern1d)
	# Normalizzazione
	return kern2d / kern2d.sum()

# Esegui l'fft del kernel K di dimensione d agggiungendo gli zeri necessari 
# ad arrivare a dimensione shape
def psf_fft(K, d, shape):
	# Aggiungi zeri
	K_p = np.zeros(shape)
	K_p[:d, :d] = K

	# Sposta elementi
	p = d // 2
	K_pr = np.roll(np.roll(K_p, -p, 0), -p, 1)

	# Esegui FFT
	K_otf = fft.fft2(K_pr)
	return K_otf

# Moltiplicazione per A
def A(x, K):
	x = fft.fft2(x)
	return np.real(fft.ifft2(K * x))

# Moltiplicazione per A trasposta
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


# load img
#X = data.camera().astype(np.float64) / 255.0 # type: ignore
X = data.cat().astype(np.float64) / 255.0 # type: ignore
#X = data.horse().astype(np.float64) / 255.0 # type: ignore

X = get_gray_image(X)

## create kernel
kernlens_sigmas = (
	(5, .5),
	(7, .1),
	(9, 1.3)
)

for (kernlen, sigma) in kernlens_sigmas:

	noise_sigma = 0.03
	ker = gaussian_kernel(kernlen, sigma)
	## create K
	K = psf_fft(ker, kernlen, X.shape)

	## apply blur
	blurred = A(X, K)

	# apply noise
	noise = np.random.normal(loc=0, scale=noise_sigma, size=X.shape)
	blurred_and_noised = blurred + noise
	blurred_and_noisedT = AT(blurred_and_noised, K)

	# peak signal noise ration, mean squared error
	psnr = metrics.peak_signal_noise_ratio(X, blurred_and_noised)
	mse = metrics.mean_squared_error(X, blurred_and_noised)


	corrupted()
	naive()
	#tikhonov()

plt.show()