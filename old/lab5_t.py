import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics
from scipy import signal
from numpy import fft
from scipy.optimize import minimize

np.random.seed(0)

""" genera filtro gaussiano / kerel : diffusione singoo punto """
# Crea un kernel Gaussiano di dimensione kernlen e deviazione standard sigma
# dev std della gaussiana
def gaussian_kernel(kernlen, sigma):
    x = np.linspace(- (kernlen // 2), kernlen // 2, kernlen)    
    # Kernel gaussiano unidmensionale
    kern1d = np.exp(- 0.5 * (x**2 / sigma))
    # Kernel gaussiano bidimensionale
    kern2d = np.outer(kern1d, kern1d)
    # Normalizzazione
    return kern2d / kern2d.sum()

""" fa padding (ZERO PADDING) : fa tutti zeri, e ci mette sopra il filtro """
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
#trasformate discrete di fourier
# x in forma matriciale
def A(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(K * x))

# Moltiplicazione per A trasposta
def AT(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(np.conj(K) * x))






#problema test

X = data.camera().astype(np.float64) / 255.0
m,n = X.shape

#genera il filtrol di blur
K = psf_fft(gaussian_kernel(9,3), 9, X.shape)

#genera noise (additivo : aggiunto all'immagine successivamente)
#anche noise preso come punti di gaussiana
#buon valore tra 1.e-3, 1.e-1
#rumore alto difficile da togliere
sigma = 0.02
#genera distrib. normale (significa gaussiana)
noise = np.random.normal(0, sigma, size=X.shape)

#aggiungi blur e noise : (k*x)+w : k convoluzione x
b = A(X,K) + noise

PSNR = metrics.peak_signal_noise_ratio(X,b)

plt.figure(figsize=(30,10))
#immagine esatta x
ax1=plt.subplot(1,2,1)
ax1.imshow(X, cmap='gray')
plt.title("Immagine originale")


ax2=plt.subplot(1,2,2)
ax2.imshow(b, cmap='gray')
plt.title(f'Immagine Corrotta (PSNR: {PSNR: .2f})', fontsize=20)

plt.show()


#PROBLEMA : SOLUZIONE NAIF

#k*x=b sse min{ax-b}2**2 sse At A x = At b
#molto grande: metodo iterativo, o gradiente per risolvere minimmo

def f(x):
    #ridimensiona per applicare A
    x_r = np.reshape(x, (m,n))
    res = (0.5)*(np.sum(np.square(A(x_r,K) - b)))
    return res

def df(x):
    x_r = np.reshape(x, (m,n))
    res = AT(A(x_r,K),K) - AT(b,K)
    res = np.reshape(res, m*n)
    return res



x0 = b
maxit = 25  #tende a essere lento

res = minimize(f, x0, method='CG', jac=df, options={'maxiter':maxit,'return_all':True} )
#ritorna immagine come vettore : per disegno va fatto reshape
print('res:', res)


res_r = np.reshape(res.x, (m,n))
PSNR_n = metrics.peak_signal_noise_ratio(X, res_r)

plt.figure(figsize=(30,10))
#immagine esatta x
ax1=plt.subplot(1,2,1)
ax1.imshow(X, cmap='gray')
plt.title("Immagine originale")


ax2=plt.subplot(1,2,2)
ax2.imshow(res_r, cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine Corrotta (PSNR: {PSNR_n: .2f})', fontsize=20)

plt.show()

#molto sfocato perché problema molto mal posto
#serve metodo di regolarizzazione
#cosi risolvo min   |ax-b|2**2 + lambda|x|2**2


#PROBLEMA : SOLUZIONE REGOLARIZZATA


lam = 0.05

def fl(x):
    #ridimensiona per applicare A
    x_r = np.reshape(x, (m,n))
    res = (0.5)*(np.sum(np.square(A(x_r,K) - b))) + lam/2 * np.sum(np.square(x_r) )
    return res

def dfl(x):
    x_r = np.reshape(x, (m,n))
    res = AT(A(x_r,K),K) - AT(b,K) + lam * x_r
    res = np.reshape(res, m*n)
    return res


x0 = b
maxit = 25  #tende a essere lento

res = minimize(fl, x0, method='CG', jac=dfl, options={'maxiter':maxit,'return_all':True} )
#ritorna immagine come vettore : per disegno va fatto reshape
print('res:', res)


res_r = np.reshape(res.x, (m,n))
PSNR_n = metrics.peak_signal_noise_ratio(X, res_r)

plt.figure(figsize=(30,10))
#immagine esatta x
ax1=plt.subplot(1,2,1)
ax1.imshow(X, cmap='gray')
plt.title("Immagine originale")


ax2=plt.subplot(1,2,2)
ax2.imshow(res_r, cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine Corrotta (PSNR: {PSNR_n: .2f})', fontsize=20)

plt.show()

#provare altri lambda per vedere quale migliora meglio



#provare maxit piu alti
#infatti c'è scirtto success=false: significa fermato per numero iterazioni, quindi in realta
#non ho ottenuto soluzione vera
#100-150 it (dovrebbe arrivare a convergenza)

#plottare errore - lambda

#plottare errore psnr - iterazioni

#oss prima di arrivare a convergenza : prima migliora poi peggiora


#cambia immagine (no cameraman!!)

""""
  plt.imread : ritorna matrice interi : fare cast a float
  """