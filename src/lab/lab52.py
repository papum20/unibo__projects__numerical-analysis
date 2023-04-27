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
import scipy
from scipy import signal
from scipy.optimize import minimize

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




# image from lab51
FIG_SHAPE = [2,2,1]
fig = plt.figure(figsize=Constants.FIGSIZE)
# load img
camera = data.camera().astype(np.float64) / 255.0
prints.img(camera, FIG_SHAPE)

# apply blur
kernlen = 24
sigma = 3
ker = gaussian_kernel(kernlen, sigma)
K = psf_fft(ker, kernlen, camera.shape)
blurred = A(camera, K)
prints.img(blurred, FIG_SHAPE)

# apply noise
dev_std = 0.02
noise = np.random.normal(scale=(np.ones(camera.shape)*dev_std))
noised = A(blurred, noise)
prints.img(noised, FIG_SHAPE)

print(f'camera: {camera.shape}, blurred: {blurred.shape}, noised: {noised.shape}')


# Peak Signal Noise Ratio
psnr = metrics.peak_signal_noise_ratio(camera, noised)
# Mean Squared Error
mse = metrics.mean_squared_error(camera, noised)

print(f'peak signal noise ration (PSNR) = {psnr},\nmean squared error (MSE) = {mse}')


# SOLUZIONE NAIVE
fmin = lambda A,b,x: (scipy.linalg.norm(np.dot(A,x) - b, ord=2) ** 2) / 2
dfmin = lambda A,b,x: A.T@A@x - np.dot(A.T, b)
f = lambda x: fmin(camera.reshape((camera.size,)).astype(np.float32), noised.reshape((noised.size,)).astype(np.float32), x)
df = lambda x: dfmin(np.atleast_2d(camera.reshape((camera.size,)).astype(np.float32)), noised.reshape((noised.size,)).astype(np.float32), x)
x0 = np.ones(camera.size, dtype=np.float32)
print(df(x0))
naive = minimize(f, x0, jac=df, method='Newton-CG')

print(f'result(naive): {naive.shape}')
prints.img(naive, FIG_SHAPE)
plt.show()