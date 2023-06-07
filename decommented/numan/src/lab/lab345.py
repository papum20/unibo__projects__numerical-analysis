import sys
sys.path.append("lib")

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from skimage import data
from numan import (
	matrix,
)



FIGSIZE		= (15,7)
FONTSIZE	= 5
STEPS		= 300
PLOTSHAPE	= (4, 4)



def analyzeImg(A, p_min, p_max, p_step, plot_shape=PLOTSHAPE):

	A_p		= {}
	err_r	= {}
	c		= {}

	(U, s, VT) = scipy.linalg.svd(A)
	for p in range(p_min, p_max, p_step):
		p		= int(p)
		A_t		= sum([np.outer(U[:,i], VT[i]*s[i]) for i in range(p)])
		A_p[p]	= (A_t)
		err_r[p] = (matrix.errRel(A, A_p[p]))
		c[p]	= (min(A_p[p].shape) / p - 1)

	x_plot = np.arange(p_min, p_max, p_step)


	print(type(A))
	print(A.shape)
	plt.imshow(A, cmap='gray')
	plt.show()
	print('\n')

	print('errori relativi:')
	for (k, v) in err_r.items():
		print("p = ", k, " : ", v)
	print('fattori di compressione c:')
	for (k, v) in c.items():
		print("p = ", k, " : ", v)


	plt.figure(figsize=FIGSIZE)
	plt.rc("font", size=FONTSIZE)

	#immagine vera
	fig = plt.subplot(plot_shape[0], plot_shape[1], 1)
	fig.imshow(A, cmap='gray')
	plt.title('True image')
	#plot err
	fig = plt.subplot(plot_shape[0], plot_shape[1], 2)
	fig.plot(x_plot, err_r.values(), label='err_r')
	fig.legend()
	plt.title('err r.')
	#plot c
	fig = plt.subplot(plot_shape[0], plot_shape[1], 3)
	fig.plot(x_plot, c.values(), label='c')
	fig.legend()
	plt.title('c')

	#immagini approssimate
	i = 0
	for (p,val) in A_p.items():
		fig = plt.subplot(plot_shape[0], plot_shape[1], 3+1+i)
		fig.imshow(val, cmap='gray')
		plt.title('Reconstructed image with p = ' + str(p))
		i += 1

	plt.show()



""" 4 """

A = data.coins()
p_min, p_max, p_step = (1, 10+1, 1)

print("\n\nCOINS 1\n")
analyzeImg(A, p_min, p_max, p_step)

""" 5 """

#coins con piu p
A = data.coins()
p_min, p_max, p_step = (1, 22, 1)

print("\n\nCOINS 2\n")
analyzeImg(A, p_min, p_max, p_step, (5,5))

#camera
A = data.camera()
p_min, p_max, p_step = (1, 10+1, 1)

print("\n\CAMERA 1\n")
analyzeImg(A, p_min, p_max, p_step)

#

A = data.horse()
p_min, p_max, p_step = (1, 16, 3)

print("\n\MITOSIS 1\n")
analyzeImg(A, p_min, p_max, p_step)