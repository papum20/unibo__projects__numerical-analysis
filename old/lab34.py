import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from skimage import data

from numan import (
	matrix as mat,
	polynom as pol
)



# c = coordinate as list
# size = [width, height]
def coordinate_next(c, size):
	if c[0] + 1 < size[0]: return [c[0]+1, c[1]]
	else: return [0, c[1]+1]
def coordinate_index(coordinate, size):
	return coordinate[1]*size[0] + coordinate[0]


##########	4	##########


# A = data.camera()
A = data.coins()
p_min = 1
p_max = 10+1
p_step = 1
plot_shape = (4, 4)
plot_size = [i*4 for i in plot_shape]
plot_font_title = 5

print(type(A))
print(A.shape)
plt.imshow(A, cmap='gray')
plt.show()

A_p = []
err_rel = []
c = []

(U, s, VT) = scipy.linalg.svd(A)
for p in range(p_min, p_max):
	A_p.append(sum([np.outer(U[:,i], VT[i]*s[i]) for i in range(p)]))
	err_rel.append(scipy.linalg.norm(A - A_p[p-p_min], 2))
	c.append(min(A_p[p-p_min].shape) / p - 1)				# fattore di compressione


# DRAW

x_plot = np.arange(p_min, p_max, p_step)
#	coordinate = [0, 0]
plt.figure(figsize=plot_size)

#immagine vera
fig = plt.subplot(plot_shape[0], plot_shape[1], 1)
fig.imshow(A, cmap='gray')
plt.title('True image', fontsize=plot_font_title)
#plot err
fig = plt.subplot(plot_shape[0], plot_shape[1], 2)
fig.plot(x_plot, c, label='c')
fig.legend()
plt.title('True image',fontsize=plot_font_title)
#plot c
fig = plt.subplot(plot_shape[0], plot_shape[1], 3)
fig.plot(x_plot, c, label='c')
fig.legend()
plt.title('True image',fontsize=plot_font_title)

#immagini approssimate
for p in range(p_min,p_max):
	fig = plt.subplot(plot_shape[0], plot_shape[1], 3+1+p-p_min)
	fig.imshow(A_p[p-p_min], cmap='gray')
	plt.title('Reconstructed image with p = ' + str(p), fontsize=plot_font_title)

plt.show()
