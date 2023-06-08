# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:34:02 2023

@author: giuse
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics
from numpy import fft
from scipy.optimize import minimize
from skimage.metrics import mean_squared_error

np.random.seed(0)

def convert_image_to_gray(m,n,H):
    X=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            X[i][j]=H[i][j][0]
    return X


def gaussian_kernel(kernlen, sigma): 
                                     
    x = np.linspace(- (kernlen // 2), kernlen // 2, kernlen)    
    
    kern1d = np.exp(- 0.5 * (x**2 / sigma))
    
    kern2d = np.outer(kern1d, kern1d)
   
    return kern2d / kern2d.sum()


def psf_fft(K, d, shape):

    K_p = np.zeros(shape)
    K_p[:d, :d] = K

    p = d // 2
    K_pr = np.roll(np.roll(K_p, -p, 0), -p, 1)
    
    K_otf = fft.fft2(K_pr)
    return K_otf


def A(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(K * x))


def AT(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(np.conj(K) * x))


H = data.astronaut().astype(np.float64)/255.0 
m, n, c = H.shape
X=convert_image_to_gray(m,n,H)



K = psf_fft(gaussian_kernel(24,3),24,X.shape)


sigma_1= 0.01
sigma_2 = 0.05
sigma_3 = 0.1
noise_1 = np.random.normal(0, sigma_1, size= X.shape)       
noise_2 = np.random.normal(0, sigma_2, size= X.shape)
noise_3 = np.random.normal(0, sigma_3, size= X.shape)

b = A(X,K) + noise_1      
b_2 = A(X,K) + noise_2
b_3 = A(X,K) + noise_3            
PSNR = metrics.peak_signal_noise_ratio(X, b)
MSE = mean_squared_error(X, b)
PSNR_2 = metrics.peak_signal_noise_ratio(X, b_2)
MSE_2 = mean_squared_error(X, b_2)
PSNR_3 = metrics.peak_signal_noise_ratio(X, b_3)  
MSE_3 = mean_squared_error(X, b_3)


def draw_image(b,X,sigma, PSNR,i,MSE, si=False):
    if si:
        plt.subplot(2,2,i)
        plt.imshow(X, cmap='gray')
        plt.title('Immagine originale', fontsize=40)
    else:
        plt.subplot(2,2,i)
        plt.imshow(b, cmap='gray')
        plt.title(f'Immagine Corrotta PSNR: {PSNR:.2f} MSE: {MSE:.4f} $\sigma$ = '+ str(sigma), fontsize= 40)

plt.figure(figsize=(40,20))
draw_image(b,X,sigma_1,PSNR,1,MSE, True)   
draw_image(b,X,sigma_1,PSNR,2,MSE) 
draw_image(b_2,X,sigma_2,PSNR_2,3,MSE_2) 
draw_image(b_3,X,sigma_3,PSNR_3,4,MSE_3)  
plt.show()



def f(x):
    x_r = np.reshape(x,(m,n))  
    res = (0.5)*(np.sum(np.square((A(x_r,K)-b))))    
                     
    return res

def df(x):
    x_r = np.reshape(x,(m,n))
    res=AT(A(x_r,K),K)-AT(b,K)   
    res = np.reshape(res, m*n)
    return res

max_it = 5 
max_it_1 = 13



def draw_naive(sigma,b,PSNR,MSE):
    res_naive_1 = minimize(f, b, method= 'CG', jac=df, options={'maxiter':max_it,'return_all':True})
    res_n_naive_1 = np.reshape(res_naive_1.x,(m,n))     
    PSNR_n_naive_1= metrics.peak_signal_noise_ratio(X, res_n_naive_1)
    MSE_naive_1 = mean_squared_error(X, res_n_naive_1)
    
    res_naive_2 = minimize(f, b, method= 'CG', jac=df, options={'maxiter':max_it_1,'return_all':True})
    res_n_naive_2 = np.reshape(res_naive_2.x,(m,n))     
    PSNR_n_naive_2= metrics.peak_signal_noise_ratio(X, res_n_naive_2)
    MSE_naive_2 = mean_squared_error(X, res_n_naive_2)

    plt.figure(figsize=(60,18))
    plt.subplot(1,3,1)
    plt.imshow(b, cmap='gray')
    plt.title(f'Immagine Corrotta PSNR:{PSNR: .2f} MSE: {MSE:.4f} $\sigma$ = '+ str(sigma), fontsize= 45)

    plt.subplot(1,3,2)
    plt.imshow(res_n_naive_1, cmap='gray')
    plt.title(f'PSNR:{PSNR_n_naive_1: .2f} MSE: {MSE_naive_1:.4f} con maxite = ' + str(max_it) ,fontsize=45)
    plt.suptitle("SOLUZIONE NAIVE", fontsize = 60)
    
    plt.subplot(1,3,3)
    plt.imshow(res_n_naive_2, cmap='gray')
    plt.title(f'PSNR:{PSNR_n_naive_2: .2f} MSE: {MSE_naive_2:.4f} con maxite = ' + str(max_it_1),fontsize=45)
    
    plt.show()



draw_naive(sigma_1,b,PSNR,MSE)   
draw_naive(sigma_2,b_2,PSNR_2,MSE_2)
draw_naive(sigma_3,b_3,PSNR_3,MSE_3)  



lam = np.array([0.0001, 0.01, 0.1]) 

def f_1(x, lam_singolo):
    x_r = np.reshape(x,(m,n))    
    res = (0.5)*(np.sum(np.square((A(x_r,K)-b)))) + (lam_singolo * 0.5)*(np.sum(np.square(x_r)))
    return res

def df_1(x, lam_singolo):
    x_r = np.reshape(x,(m,n))
    res=AT(A(x_r,K),K)-AT(b,K) + lam_singolo*x_r  
    res = np.reshape(res, m*n) 
    return res



def draw_regolarized(i, lam, max_ite,sigma,b,PSNR,MSE, SiOriginale=False):
    res_regolarized = minimize(f_1, b, lam, method= 'CG', jac=df_1, options={'maxiter':max_ite,'return_all':True})
    
    res_n_regolarized = np.reshape(res_regolarized.x,(m,n))     
    PSNR_n_regolarized = metrics.peak_signal_noise_ratio(X, res_n_regolarized)
    MSE_reg = mean_squared_error(X, res_n_regolarized)

    if(SiOriginale):
        plt.subplot(2,2,i-1)
        plt.imshow(b, cmap='gray')
        plt.title(f'Immagine Corrotta PSNR:{PSNR: .4f} MSE: {MSE:.4f} $\sigma$ = '+ str(sigma), fontsize= 40)
        plt.suptitle("immagine regolarizzata con maxIterazioni = " + str(max_ite),fontsize=60)

    plt.subplot(2,2,i)
    plt.imshow(res_n_regolarized, cmap='gray')
    plt.title(f'Immagine Ricostruita PSNR: {PSNR_n_regolarized:.4f} MSE: {MSE_reg:.4f} $\lambda$ = '+ str(lam), fontsize= 40)


plt.figure(figsize=(40,20))
draw_regolarized(2,lam[0],max_it,sigma_1,b,PSNR,MSE, True)
draw_regolarized(3,lam[1],max_it,sigma_1,b,PSNR,MSE)
draw_regolarized(4,lam[2],max_it,sigma_1,b,PSNR,MSE)
plt.show() 

    
plt.figure(figsize=(40,20))
draw_regolarized(2,lam[0],max_it_1,sigma_1,b,PSNR,MSE, True)
draw_regolarized(3,lam[1],max_it_1,sigma_1,b,PSNR,MSE)
draw_regolarized(4,lam[2],max_it_1,sigma_1,b,PSNR,MSE)
plt.show()           
    
plt.figure(figsize=(40,20))
draw_regolarized(2,lam[0],max_it,sigma_2,b_2,PSNR_2,MSE_2, True)
draw_regolarized(3,lam[1],max_it,sigma_2,b_2,PSNR_2,MSE_2)
draw_regolarized(4,lam[2],max_it,sigma_2,b_2,PSNR_2,MSE_2)
plt.show()   

    
plt.figure(figsize=(40,20))
draw_regolarized(2,lam[0],max_it_1,sigma_2,b_2,PSNR_2,MSE_2, True)
draw_regolarized(3,lam[1],max_it_1,sigma_2,b_2,PSNR_2,MSE_2)
draw_regolarized(4,lam[2],max_it_1,sigma_2,b_2,PSNR_2,MSE_2)
plt.show()      

plt.figure(figsize=(40,20))
draw_regolarized(2,lam[0],max_it,sigma_3,b_3,PSNR_3,MSE_3, True)
draw_regolarized(3,lam[1],max_it,sigma_3,b_3,PSNR_3,MSE_3)
draw_regolarized(4,lam[2],max_it,sigma_3,b_3,PSNR_3,MSE_3)
plt.show()   

    
plt.figure(figsize=(40,20))
draw_regolarized(2,lam[0],max_it_1,sigma_3,b_3,PSNR_3,MSE_3, True)
draw_regolarized(3,lam[1],max_it_1,sigma_3,b_3,PSNR_3,MSE_3)
draw_regolarized(4,lam[2],max_it_1,sigma_3,b_3,PSNR_3,MSE_3)
plt.show()       
    

ite = 13

PSNR_plot_l1 = np.zeros(ite)
MSE_plot_l1 = np.zeros(ite)
PSNR_plot_l2 = np.zeros(ite)
MSE_plot_l2 = np.zeros(ite)
PSNR_plot_l3 = np.zeros(ite)
MSE_plot_l3 = np.zeros(ite)

for i in range (ite):
    res_regolarized_l1 = minimize(f_1, b_3, lam[0], method= 'CG', jac=df_1, options={'maxiter':i,'return_all':True})
    res_n_regolarized_l1 = np.reshape(res_regolarized_l1.x,(m,n))     
    PSNR_plot_l1[i] = metrics.peak_signal_noise_ratio(X, res_n_regolarized_l1)
    MSE_plot_l1[i] = mean_squared_error(X, res_n_regolarized_l1)
    
    res_regolarized_l2 = minimize(f_1, b_3, lam[1], method= 'CG', jac=df_1, options={'maxiter':i,'return_all':True})
    res_n_regolarized_l2 = np.reshape(res_regolarized_l2.x,(m,n))     
    PSNR_plot_l2[i] = metrics.peak_signal_noise_ratio(X, res_n_regolarized_l2)
    MSE_plot_l2[i] = mean_squared_error(X, res_n_regolarized_l2)
    
    res_regolarized_l3 = minimize(f_1, b_3, lam[2], method= 'CG', jac=df_1, options={'maxiter':i,'return_all':True})
    res_n_regolarized_l3 = np.reshape(res_regolarized_l3.x,(m,n))     
    PSNR_plot_l3[i] = metrics.peak_signal_noise_ratio(X, res_n_regolarized_l3)
    MSE_plot_l3[i] = mean_squared_error(X, res_n_regolarized_l3)
    
ite_plot = np.arange(ite) 


plt.plot(ite_plot, PSNR_plot_l1, color = 'blue')
plt.plot(ite_plot, PSNR_plot_l2, color = 'yellow')
plt.plot(ite_plot, PSNR_plot_l3, color = 'red')
plt.legend(['$\lambda$ = ' + str(lam[0]),'$\lambda$ = ' + str(lam[1]),'$\lambda$ = ' + str(lam[2])]) 
plt.title('PSNR al variare di lambda e le iterazioni')
plt.xlabel('iterazioni')
plt.show()


plt.plot(ite_plot, MSE_plot_l1, color = 'blue')
plt.plot(ite_plot, MSE_plot_l2, color = 'yellow')
plt.plot(ite_plot, MSE_plot_l3, color = 'red')
plt.title('MSE al variare di lambda e le iterazioni')
plt.legend(['$\lambda$ = ' + str(lam[0]),'$\lambda$ = ' + str(lam[1]),'$\lambda$ = ' + str(lam[2])]) 
plt.xlabel('iterazioni')
plt.show()