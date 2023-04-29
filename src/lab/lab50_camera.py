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




# load img
camera = data.camera().astype(np.float64) / 255.0

## create kernel
kernlen = 24
sigma = 3
ker = gaussian_kernel(kernlen, sigma)
## create K
K = psf_fft(ker, kernlen, camera.shape)

## apply blur
blurred = A(camera, K)

# apply noise
dev_std = 0.02
noise = np.random.normal(scale=np.ones(camera.shape)) * dev_std
noised = blurred + noise
noisedT = AT(noised, K)

# peak signal noise ration, mean squared error
psnr = metrics.peak_signal_noise_ratio(camera, noised)
mse = metrics.mean_squared_error(camera, noised)

