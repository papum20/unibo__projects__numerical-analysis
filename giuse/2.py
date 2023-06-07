# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 21:56:07 2023

@author: giuse
"""

import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt


start=10
finish=100

K_H = np.zeros(13)
Err_H = np.zeros(13)

for n in np.arange(2,15):
    
    H = scipy.linalg.hilbert(n) 
    x_true_h = np.ones((n,1))
    b_h = H@x_true_h


    K_H[n-2] = np.linalg.cond(H, 2)

   
    L_h = scipy.linalg.cholesky (H)  
    y_h = np.linalg.solve(L_h,b_h)
    my_x_h = np.linalg.solve(L_h.T,y_h)
    Err_H[n-2] = np.linalg.norm((my_x_h-x_true_h), 2)/np.linalg.norm(x_true_h, 2)


n_h=np.arange(2,15)

plt.loglog(n_h, K_H)
plt.title('CONDIZIONAMENTO DI H ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(H)')
plt.show()

plt.loglog(n_h, Err_H)
plt.title('Errore relativo H')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err_H= ||my_x-x||/||x||')
plt.show()
