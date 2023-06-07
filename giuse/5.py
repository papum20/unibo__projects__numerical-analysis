# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:50:19 2023

@author: giuse
"""

import numpy as np
import scipy
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec 
import matplotlib.pyplot as plt
import time 


def Jacobi(A,b,x0,maxit,tol, xTrue):
  n=np.size(x0)     
  ite=0
  x = np.copy(x0)  
  norma_it=1+tol 
  relErr=np.zeros((maxit, 1))
  errIter=np.zeros((maxit, 1))
  relErr[0]=np.linalg.norm(xTrue-x0)/np.linalg.norm(xTrue)
  while (ite<maxit and norma_it>tol):
    x_old=np.copy(x) 
    for i in range(0,n): 
      x[i]=(b[i]-np.dot(A[i,0:i],x_old[0:i])-np.dot(A[i,i+1:n],x_old[i+1:n]))/A[i,i] 
    ite=ite+1
    relErr[ite-1] = np.linalg.norm(xTrue-x)/np.linalg.norm(xTrue)
    norma_it = np.linalg.norm(x_old-x)/np.linalg.norm(x_old)
    errIter[ite-1] = norma_it 
  relErr=relErr[:ite]
  errIter=errIter[:ite]  
  return [x, ite, relErr, errIter]


def GaussSeidel(A,b,x0,maxit,tol, xTrue):
    n=np.size(x0)     
    ite=0
    x = np.copy(x0) 
    norma_it=1+tol 
    relErr=np.zeros((maxit, 1))
    errIter=np.zeros((maxit, 1))
    relErr[0]=np.linalg.norm(xTrue-x0)/np.linalg.norm(xTrue)
    while (ite<maxit and norma_it>tol):
        x_old=np.copy(x)  
        for i in range(0,n): 
            x[i]=(b[i]-np.dot(A[i,0:i],x[0:i])-np.dot(A[i,i+1:n],x_old[i+1:n]))/A[i,i]  
        ite=ite+1
        relErr[ite-1] = np.linalg.norm(xTrue-x)/np.linalg.norm(xTrue)
        norma_it = np.linalg.norm(x_old-x)/np.linalg.norm(x_old)
        errIter[ite-1] = norma_it 
        
    relErr=relErr[:ite]
    errIter=errIter[:ite]  
    return [x, ite, relErr, errIter]

    
def time_function(A, b, x0, x_true, choice):
    if choice==1:
        start_time=time.time()
        x, empty, relErr, empty = GaussSeidel(A,b,x0,maxit,tol, x_true)
        Err = np.linalg.norm(x_true-x)/np.linalg.norm(x_true) 
    elif choice == 2:
        start_time=time.time()
        x, empty, relErr, empty = Jacobi(A,b,x0,maxit,tol, x_true)
        Err = np.linalg.norm(x_true-x)/np.linalg.norm(x_true) 
    elif choice == 3:
        start_time=time.time()
        lu, piv =LUdec.lu_factor(A) 
        my_x=scipy.linalg.lu_solve((lu,piv), b)
        Err = np.linalg.norm((my_x-x_true), 2)/np.linalg.norm(x_true, 2)
    elif choice == 4:
        start_time=time.time()
        L = scipy.linalg.cholesky (A, lower=True)
        y = np.linalg.solve(L,b)
        my_x = np.linalg.solve(L.T,y)
        Err = np.linalg.norm((my_x-x_true), 2)/np.linalg.norm(x_true, 2)
    return (time.time()-start_time), Err


D_val = 9
D_val_u = -4
D_val_d = -4


maxit = 300
tol = 1.e-8

dim = np.arange(10,110)

i = 0


err_j=np.zeros(np.size(dim))
err_gs=np.zeros(np.size(dim))
err_LU=np.zeros(np.size(dim))
err_CH=np.zeros(np.size(dim))

time_j=np.zeros(np.size(dim))
time_gs=np.zeros(np.size(dim))
time_LU=np.zeros(np.size(dim))
time_CH=np.zeros(np.size(dim))

for n in dim:
    
    x0 = np.zeros((n,1))
    x0[0]=1
    
    D = np.eye(n,k=0)*D_val + np.eye(n,k=1)*D_val_u + np.eye(n,k=-1)*D_val_d 
    x_true_d=np.ones((n,1))
    b_d = D@x_true_d
    
    time_gs[i], err_gs[i] = time_function(D,b_d,x0,x_true_d,1)
    time_j[i], err_j[i] = time_function(D,b_d,x0,x_true_d,2)
    time_LU[i], err_LU[i] = time_function(D,b_d,x0,x_true_d,3)
    time_CH[i], err_CH[i] = time_function(D,b_d,x0,x_true_d,4)
    
    i = i + 1
    
    
plt.figure(figsize=(20,15))

plt.plot(dim,err_j, color ='red')
plt.plot(dim,err_gs, color ='yellow')
plt.plot(dim,err_LU, color ='blue')
plt.plot(dim,err_CH, color ='green')
plt.legend(['JACOBI','GAUSS SIDELL','LU','Cholesky'])
plt.xlabel('dimensione matrice: n')
plt.ylabel('errore relativo')

plt.show()

plt.figure(figsize=(20,15))

plt.plot(dim,time_j, color ='red')
plt.plot(dim,time_gs, color ='yellow')
plt.plot(dim,time_LU, color ='blue')
plt.plot(dim,time_CH, color ='green')
plt.legend(['JACOBI','GAUSS SIDELL','LU','Cholesky'])
plt.xlabel('dimensione matrice: n')
plt.ylabel('tempo')

plt.show()