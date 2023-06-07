# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 21:34:14 2023

@author: giuse
"""

import numpy as np
import scipy
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec 
import matplotlib.pyplot as plt


start=10
finish=1000

K_R = np.zeros(finish-start)
Err_R = np.zeros(finish-start)

k = 0
for n in np.arange(start,finish):


    R = np.random.randint(start, finish, size=(n,n))
    x_true_r = np.ones((n, 1))
    b_r=R@x_true_r

    lu, piv =LUdec.lu_factor(R) 

    my_x_r=scipy.linalg.lu_solve((lu,piv), b_r)
    K_R[n-start] = np.linalg.cond(R, 2)
    Err_R[n-start] = np.linalg.norm((my_x_r-x_true_r), 2)/np.linalg.norm(x_true_r, 2)
   


n = np.arange(start,finish) 

plt.loglog(n, K_R)
plt.title('CONDIZIONAMENTO DI R ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(R)')
plt.show()

plt.loglog(n, Err_R)
plt.title('Errore relativo R')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err_R= ||my_x-x||/||x||')
plt.show()
