import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from numan import (
	matrix as mat,
	polynom as pol
)


##########	4	##########


from skimage import data


# A = data.camera()
A = data.coins()

print(type(A))
print(A.shape)


plt.imshow(A, cmap='gray')
plt.show()


...

...

A_p = np.zeros(A.shape)
p_max = 10


for i in range(p_max):
    
  ...

err_rel = ...
c = ...

print('\n')
print('L\'errore relativo della ricostruzione di A è', err_rel)
print('Il fattore di compressione è c=', c)


plt.figure(figsize=(20, 10))

fig1 = plt.subplot(1, 2, 1)
fig1.imshow(A, cmap='gray')
plt.title('True image')

fig2 = plt.subplot(1, 2, 2)
fig2.imshow(A_p, cmap='gray')
plt.title('Reconstructed image with p =' + str(p_max))

plt.show()



# al variare di p
