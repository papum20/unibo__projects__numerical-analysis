# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:08:24 2023

@author: giuse
"""

import numpy as np
import math
import matplotlib.pyplot as plt


def cut(vect, finish):
    new_vect=np.zeros((finish,1))
    for i in range(finish):
        new_vect[i]=vect[i]
    return new_vect

def bisezione(a, b, f, tolx, xTrue): 

  k = math.ceil(math.log(abs(b-a)/tolx)/math.log(2))                                   
  vecErrore = np.zeros( (k,1) )             
  if f(a)*f(b)>0: 
    print("nell'intervallo [a,b] non ci sono intersezioni con l'asse delle x")
    return 
  for i in range(1,k+1):
      c= a + (b-a)/2      
      vecErrore[i-1]=abs (c-xTrue)
      if abs(f(c))<tolx:          
          new_vect = cut(vecErrore,i)
          return c, i, k, new_vect
      else:          
        if f(a)*f(c)<0:
          b=c
        else:
          a=c
      x=c 
  return (x, i, k, vecErrore)

      
''' Metodo di Newton'''

def newton(f, df, tolf, tolx, maxit, xTrue, x0=0):
  
  err=np.zeros(maxit, dtype=float)
  vecErrore=np.zeros( (maxit,1), dtype=float)
  i=0
  err[0]=tolx+1
  vecErrore[0] = np.abs(x0-xTrue)
  x=x0
  while (abs(f(x))>tolf and i<maxit-1): 
    x_new=x-(f(x)/df(x))
    err[i+1] = abs(x_new-x)
    vecErrore[i+1] = abs (x_new-xTrue) 
    i=i+1
    x=x_new
  new_vect1 = cut(vecErrore,i)
  new_vect2 = cut(err,i)
  return (x, i, new_vect2, new_vect1)  

def approssimazioni_successive(f, g, tolf, tolx, maxit, xTrue, x0=0):
  err=np.zeros(maxit + 1, dtype=float)
  vecErrore=np.zeros( (maxit + 1,1), dtype=float)
  i=0
  err[0]=tolx+1
  vecErrore[0] = abs(x0-xTrue)
  x=x0
  while (abs(f(x))>tolf and err[i]>tolx and i<maxit-1): 
    x0 = x
    x = g(x0)
    i=i+1
    err[i] = abs(x -x0)
    vecErrore[i] = abs (x-xTrue) 
  new_vect1 = cut(vecErrore,i)
  new_vect2 = cut(err,i)
  return (x, i, new_vect2, new_vect1)  

f = lambda x: (x**3 + 4*x*np.cos(x)-2) #intervallo [0,2]
df = lambda x: (3*x**2 + 4*np.cos(x)-4*x*np.sin(x))
g = lambda x: ((2-x**3)/(4*np.cos(x))) 


a=0
b=2


xTrue = 0.536839
tolx= 10**(-10)
tolf = 10**(-9)
maxit=200
x0= tolx
x_plot = np.linspace(a, b, 101)

[x, ite_bisezione, k, vecErrore_bisezione] = bisezione(a,b,f,tolx, xTrue)
x, ite_newton, err, vecErrore_newton = newton(f,df,tolf,tolx,maxit,xTrue,x0)
[x, ite_g, err_g, vecErrore_g] = approssimazioni_successive(f,g,tolf,tolx,maxit,xTrue,x0)


ite_b = np.arange(ite_bisezione)
ite_n = np.arange(ite_newton)
ite_g = np.arange(ite_g)


plt.figure(figsize=(20,15))
plt.loglog(ite_b, vecErrore_bisezione, color = 'blue')
plt.loglog(ite_n, vecErrore_newton, color = 'yellow')
plt.loglog(ite_g, vecErrore_g, color = 'red')
plt.legend(['bisezione','Newton','g(x) = (2-x^3)/(4cos(x))']) 
plt.title('ITERAZIONI vs ERRORE in intervallo [0,2]')
plt.xlabel('Iterazioni')
plt.ylabel('Errore')
plt.show()

































