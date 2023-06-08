# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:16:45 2023

@author: giuse
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from skimage import data


def compressione(A,p_max):
    tmp=A.shape
    n=tmp[0]
    m=tmp[1]
    
    U, s, Vh = scipy.linalg.svd(A)
    
    A_p=np.zeros(A.shape)
    
    for i in range(p_max):
        ui=U[:,i]
        vi=Vh[i,:]
        A_p = A_p + (np.outer(ui,vi)*s[i])
    err_rel = (np.linalg.norm (A-A_p, 2))/np.linalg.norm (A, 2)
    c = ((min(m,n))/p_max)-1
    return A_p,err_rel,c


A = data.coins()

print("il rango di A è " + str(np.linalg.matrix_rank(A)))

p_max_1 = 10
p_max_2 = 30    
p_max_3 = 100
p_max_4 = 150
p_max_5 = 200

A_p1,err_rel_1, c_1 = compressione(A, p_max_1)
A_p2,err_rel_2, c_2 = compressione(A, p_max_2)
A_p3,err_rel_3, c_3 = compressione(A, p_max_3)
A_p4,err_rel_4, c_4 = compressione(A, p_max_4)
A_p5,err_rel_5, c_5 = compressione(A, p_max_5)



plt.figure(figsize=(20, 15))

fig1 = plt.subplot(3, 3, 1)
fig1.imshow(A, cmap='gray')
plt.title('True image')

fig2 = plt.subplot(3, 3, 2)
fig2.imshow(A_p1, cmap='gray')
plt.title('Reconstructed image with p =' + str(p_max_1))

fig2 = plt.subplot(3, 3, 3)
fig2.imshow(A_p2, cmap='gray')
plt.title('Reconstructed image with p =' + str(p_max_2))

fig2 = plt.subplot(3, 3, 4)
fig2.imshow(A_p3, cmap='gray')
plt.title('Reconstructed image with p =' + str(p_max_3))

fig2 = plt.subplot(3, 3, 5)
fig2.imshow(A_p4, cmap='gray')
plt.title('Reconstructed image with p =' + str(p_max_4))

fig2 = plt.subplot(3, 3, 6)
fig2.imshow(A_p5, cmap='gray')
plt.title('Reconstructed image with p =' + str(p_max_5))

plt.show()



# al variare di p
p_max=np.array([p_max_1,p_max_2,p_max_3,p_max_4,p_max_5])
err_rel=np.array([err_rel_1,err_rel_2,err_rel_3,err_rel_4,err_rel_5])
c=np.array([c_1,c_2,c_3,c_4,c_5])
print("fattore di compressione per ogni p = ",c)
print("errore relativo per ogni p = ",err_rel)

plt.figure()
plt.plot(p_max, err_rel, label=('errore relativo'), color='blue', linewidth=1, marker='.')
plt.plot(p_max, c, label='fattore di compressione', color = 'red', linewidth=1, marker='.' )
plt.legend(loc='upper right')
plt.xlabel('p')
plt.ylabel('')
plt.title('errore relativo e il fattore di compressione al variare di p')
plt.show()
