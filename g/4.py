# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 22:02:13 2023

@author: giuse
"""

import numpy as np
import matplotlib.pyplot as plt

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


D_val = 9
D_val_u = -4
D_val_d = -4

maxit = 200

tol_1 = 1.e-10
tol_2 = 1.e-9
tol_3 = 1.e-8

dim = np.arange(10,110)

ite_j1_first=np.zeros(np.size(dim))
ite_gs1_first=np.zeros(np.size(dim))
ite_j2_first=np.zeros(np.size(dim))
ite_gs2_first=np.zeros(np.size(dim))
ite_j3_first=np.zeros(np.size(dim))
ite_gs3_first=np.zeros(np.size(dim))

ite_j1_second=np.zeros(np.size(dim))
ite_gs1_second=np.zeros(np.size(dim))
ite_j2_second=np.zeros(np.size(dim))
ite_gs2_second=np.zeros(np.size(dim))
ite_j3_second=np.zeros(np.size(dim))
ite_gs3_second=np.zeros(np.size(dim))

err_j_1_first=np.zeros(np.size(dim))
err_j_2_first=np.zeros(np.size(dim))
err_j_3_first=np.zeros(np.size(dim))
err_gs_1_first=np.zeros(np.size(dim))
err_gs_2_first=np.zeros(np.size(dim))
err_gs_3_first=np.zeros(np.size(dim))

err_j_1_second=np.zeros(np.size(dim))
err_j_2_second=np.zeros(np.size(dim))
err_j_3_second=np.zeros(np.size(dim))
err_gs_1_second=np.zeros(np.size(dim))
err_gs_2_second=np.zeros(np.size(dim))
err_gs_3_second=np.zeros(np.size(dim))

i = 0
for n in dim:
    
    x0_1 = np.zeros((n,1))
    x0_1 = np.full((n,1), 0.001)
    x0_2 = np.zeros((n,1))
    x0_2 = np.full((n,1), 0.9)
    
    
    D = np.eye(n,k=0)*D_val + np.eye(n,k=1)*D_val_u + np.eye(n,k=-1)*D_val_d 
    x_true_d=np.ones((n,1))
    b_d = D@x_true_d
    
    x, n_ite_gs1_first, Err_D_GS_1_first, errIter = GaussSeidel(D,b_d,x0_1,maxit,tol_1, x_true_d)
    ite_gs1_first[i] = n_ite_gs1_first
    err_gs_1_first[i] = np.linalg.norm(x_true_d-x)/np.linalg.norm(x_true_d)
    
    x, n_ite_gs1_second, Err_D_GS_1_second, errIter = GaussSeidel(D,b_d,x0_2,maxit,tol_1, x_true_d)
    ite_gs1_second[i] = n_ite_gs1_second 
    err_gs_1_second[i] = np.linalg.norm(x_true_d-x)/np.linalg.norm(x_true_d)
    
    x, n_ite_j1_first, Err_D_J_1_first, errIter = Jacobi(D,b_d,x0_1,maxit,tol_1, x_true_d)
    ite_j1_first[i] = n_ite_j1_first 
    err_j_1_first[i] = np.linalg.norm(x_true_d-x)/np.linalg.norm(x_true_d)
    
    x, n_ite_j1_second, Err_D_J_1_first, errIter = Jacobi(D,b_d,x0_2,maxit,tol_1, x_true_d)
    ite_j1_second[i] = n_ite_j1_second 
    err_j_1_second[i] = np.linalg.norm(x_true_d-x)/np.linalg.norm(x_true_d)
    
    x, n_ite_gs2_first, Err_D_GS_2, errIter = GaussSeidel(D,b_d,x0_1,maxit,tol_2, x_true_d)
    ite_gs2_first[i] = n_ite_gs2_first 
    err_gs_2_first[i] = np.linalg.norm(x_true_d-x)/np.linalg.norm(x_true_d)
    
    x, n_ite_gs2_second, Err_D_GS_2, errIter = GaussSeidel(D,b_d,x0_2,maxit,tol_2, x_true_d)
    ite_gs2_second[i] = n_ite_gs2_second 
    err_gs_2_second[i] = np.linalg.norm(x_true_d-x)/np.linalg.norm(x_true_d)
    
    x, n_ite_j2_first, Err_D_J_2, errIter = Jacobi(D,b_d,x0_1,maxit,tol_2, x_true_d)
    ite_j2_first[i] = n_ite_j2_first 
    err_j_2_first[i] = np.linalg.norm(x_true_d-x)/np.linalg.norm(x_true_d) 
    
    x, n_ite_j2_second, Err_D_J_2, errIter = Jacobi(D,b_d,x0_2,maxit,tol_2, x_true_d)
    ite_j2_second[i] = n_ite_j2_second 
    err_j_2_second[i] = np.linalg.norm(x_true_d-x)/np.linalg.norm(x_true_d)
    
    x, n_ite_gs3_first, Err_D_GS_3, errIter = GaussSeidel(D,b_d,x0_1,maxit,tol_3, x_true_d)
    ite_gs3_first[i] = n_ite_gs3_first 
    err_gs_3_first[i] = np.linalg.norm(x_true_d-x)/np.linalg.norm(x_true_d) 
    
    x, n_ite_gs3_second, Err_D_GS_3, errIter = GaussSeidel(D,b_d,x0_2,maxit,tol_3, x_true_d)
    ite_gs3_second[i] = n_ite_gs3_second 
    err_gs_3_second[i] = np.linalg.norm(x_true_d-x)/np.linalg.norm(x_true_d) 
    
    x, n_ite_j3_first, Err_D_J_3, errIter = Jacobi(D,b_d,x0_1,maxit,tol_3, x_true_d)
    ite_j3_first[i] = n_ite_j3_first
    err_j_3_first[i] = np.linalg.norm(x_true_d-x)/np.linalg.norm(x_true_d) 
    
    x, n_ite_j3_second, Err_D_J_3, errIter = Jacobi(D,b_d,x0_2,maxit,tol_3, x_true_d)
    ite_j3_second[i] = n_ite_j3_second
    err_j_3_second[i] = np.linalg.norm(x_true_d-x)/np.linalg.norm(x_true_d) 
    
    i = i + 1 






def draw_error_J(x0,Err_D_J_1,Err_D_J_2,Err_D_J_3, tol_1,tol_2,tol_3):
    plt.plot(dim,Err_D_J_1, color = 'blue')
    plt.plot(dim,Err_D_J_2, color = 'yellow')
    plt.plot(dim,Err_D_J_3, color = 'red')
    plt.legend(['tol = ' + str(tol_1),'tol = ' + str(tol_2),'tol = ' + str(tol_3)]) 
    plt.title('Jacobi con x0 = ' + str(x0) + ' e tol = ' + str(tol))
    plt.xlabel('dimensione matrice: n')
    plt.ylabel('errore relativo')
    
    
def draw_iterations_J(x0,ite_j_1,ite_j_2,ite_j_3,tol_1,tol_2,tol_3):
    
     plt.plot(dim,ite_j_1, color = 'blue')
     plt.plot(dim,ite_j_2, color = 'yellow')
     plt.plot(dim,ite_j_3, color = 'red')
     plt.legend(['tol = ' + str(tol_1),'tol = ' + str(tol_2),'tol = ' + str(tol_3)]) 
     plt.title('Jacobi con x0 = ' + str(x0) + ' e tol = ' + str(tol))
     plt.xlabel('dimensione matrice: n')
     plt.ylabel('iterazioni')
    
def draw_error_GS(x0,Err_D_GS_1,Err_D_GS_2,Err_D_GS_3, tol_1,tol_2,tol_3):

    plt.plot(dim,Err_D_GS_1, color = 'blue')
    plt.plot(dim,Err_D_GS_2, color = 'yellow')
    plt.plot(dim,Err_D_GS_3, color = 'red')
    plt.legend(['tol = ' + str(tol_1),'tol = ' + str(tol_2),'tol = ' + str(tol_3)]) 
    plt.title('Gauss Sidell con x0 = ' + str(x0) + ' e tol = ' + str(tol))
    plt.xlabel('dimensione matrice: n')
    plt.ylabel('errore relativo')    
    
    
def draw_iterations_GS(x0,ite_gs_1,ite_gs_2,ite_gs_3,tol_1,tol_2,tol_3):

    plt.plot(dim,ite_gs_1, color = 'blue')
    plt.plot(dim,ite_gs_2, color = 'yellow')
    plt.plot(dim,ite_gs_3, color = 'red')
    plt.legend(['tol = ' + str(tol_1),'tol = ' + str(tol_2),'tol = ' + str(tol_3)]) 
    plt.title('Gauss Sidell con x0 = ' + str(x0) + ' e tol = ' + str(tol))
    plt.xlabel('dimensione matrice: n')
    plt.ylabel('iterazioni')        


x0 = np.array([x0_1[0],x0_2[0]])
tol = np.array([tol_1,tol_2,tol_3])


plt.figure(figsize=(20,15))
draw_error_J(x0[0],err_j_1_first,err_j_2_first,err_j_3_first, tol[0],tol[1],tol[2])
plt.show()

plt.figure(figsize=(20,15))
draw_error_J(x0[1],err_j_1_second,err_j_2_second,err_j_3_second, tol[0],tol[1],tol[2])
plt.show()


plt.figure(figsize=(20,15))
draw_iterations_J(x0[0], ite_j1_first,ite_j2_first,ite_j3_first, tol[0],tol[1],tol[2])
plt.show()

plt.figure(figsize=(20,15))
draw_iterations_J(x0[1], ite_j1_second,ite_j2_second,ite_j3_second, tol[0],tol[1],tol[2])
plt.show()

plt.figure(figsize=(20,15))
draw_error_GS(x0[0],err_gs_1_first,err_gs_2_first,err_gs_3_first, tol[0],tol[1],tol[2])
plt.show()

plt.figure(figsize=(20,15))
draw_error_GS(x0[1],err_gs_1_second,err_gs_2_second,err_gs_3_second, tol[0],tol[1],tol[2])
plt.show()



plt.figure(figsize=(20,15))
draw_iterations_GS(x0[0], ite_gs1_first,ite_gs2_first,ite_gs3_first, tol[0],tol[1],tol[2])

plt.show()

plt.figure(figsize=(20,15))
draw_iterations_GS(x0[1], ite_gs1_first,ite_gs2_first,ite_gs3_first, tol[0],tol[1],tol[2])
plt.show()