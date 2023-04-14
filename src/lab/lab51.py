import sys
sys.path.append("lib")

from numan import (
	Constants,
	polyrn,
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




# shape=(nrows,ncols,next_index)
def drawImg(img, shape:list[int]):
	ax = plt.subplot(shape[0], shape[1], shape[2])
	ax.imshow(img, cmap='gray')
	shape[2] += 1



FIG_SHAPE = [1,3,1]
# load img
camera = data.camera().astype(np.float64) / 255.0

print(f"camera: {camera.shape}")
fig = plt.figure(figsize=Constants.FIGSIZE)
drawImg(camera, FIG_SHAPE)

# apply blur
## create kernel
kernlen = 24
sigma = 3
ker = gaussian_kernel(kernlen, sigma)
print(f"kernel: {ker.shape}")
print(ker)
## create K
K = psf_fft(ker, kernlen, camera.shape)
print(f"K: {K.shape}")
print(K)
## apply blur
blurred = A(camera, K)

drawImg(blurred, FIG_SHAPE)

# apply noise
dev_std = 0.02
noise = np.random.normal(scale=(np.ones(camera.shape)*dev_std))
print(f"noise: {noise.shape}")
print(noise)
noised = A(blurred, noise)

drawImg(noised, FIG_SHAPE)


plt.show()


# Peak Signal Noise Ratio
psnr = metrics.peak_signal_noise_ratio(camera, noised)
# Mean Squared Error
mse = metrics.mean_squared_error(camera, noised)

print(f'peak signal noise ration (PSNR) = {psnr},\nmean squared error (MSE) = {mse}')


# SOLUZIONE NAIVE
fmin = lambda A,b,x: (scipy.linalg.norm(A*x - b, ord=2) ** 2) / 2
dfmin = lambda A,b,x: A.T*A*x - A.T*b
f = lambda x: fmin(camera, noised, x)
df = lambda x: dfmin(camera, noised, x)
naive = minimize(f, df)