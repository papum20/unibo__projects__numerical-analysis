

import numpy as np
import math
import matplotlib.pyplot as plt

def cut(vect, finish):
    new_vect=np.zeros((finish,1))
    for i in range(finish):
        new_vect[i]=vect[i]
    return new_vect

''' Metodo di Bisezione'''
def bisezione(a, b, f, tolx,tolf, xTrue):
  
  k = math.ceil(math.log(abs(b-a)/tolx)/math.log(2))    
  vecErrore = np.zeros( (k,1) )
  if f(a)*f(b)>0:
     print("Non ci sono intersezioni sull'asse x")
     return(0, 0, 0, 0)
  for i in range(0,k):
    c = a + ((b - a)/2)
    vecErrore[i-1] = abs(c - xTrue)
    if abs(f(c)) < tolf:                                  
          x = c
          new_vect = cut(vecErrore,i)
          return (x, i, k, new_vect)
    else:
        if np.sign(f(a)*np.sign(f(c))) < 0:
         b = c
        else:
         a = c
    x=c

  return (x, i, k, vecErrore)

      
''' Metodo di Newton'''



def approssimazioni_successive(f, g, tolf, tolx, maxit, xTrue, x0=0):
  err=np.zeros(maxit+1, dtype=float)
  vecErrore=np.zeros( maxit+1, dtype=float)
  i=0
  err[0]=tolx+1
  vecErrore[0] = np.abs(x0-xTrue)
  x=x0
  while (abs(f(x))>tolf and  err[i]>tolx and i<maxit): 
    x_new=g(x)
    
    i=i+1
    vecErrore[i] = abs (x - xTrue) 
    err[i] = abs(x_new-x)
    
    x=x_new
    
  err = err[0:i]
  vecErrore = vecErrore[0:i]
  return (x, i, err, vecErrore)  



'''creazione del problema'''

f = lambda x: np.exp(x) - x**2
df = lambda x: np.exp(x) - 2 * x

g1 = lambda x: x - f(x) * np.exp(x/2)
g2 = lambda x: x - f(x) * np.exp(-x/2)
g3 = lambda x: x - f(x) / df(x)


xTrue = -0.7034674


a=-1.0
b=1.0
tolx= 10**(-10)
tolf = 10**(-6)
maxit=100
x0= 0

''' Grafico funzione in [a, b]'''

x_bis, ite_bisezione, k_bis, vecErrore_bisezione = bisezione(a,b,f,tolx,tolf, xTrue)


x_new, ite_newton, err_new, vecErrore_newton = approssimazioni_successive(f,g3,tolf,tolx,maxit,xTrue,x0)

[x_g1, ite_g1, err_g1, vecErrore_g1] = approssimazioni_successive(f,g1,tolf,tolx,maxit,xTrue,x0)

[x_g2, ite_g2, err_g2, vecErrore_g2] = approssimazioni_successive(f,g2,tolf,tolx,maxit,xTrue,x0)

[x_g3, ite_g3, err_g3, vecErrore_g3] = approssimazioni_successive(f,g3,tolf,tolx,maxit,xTrue,x0)



''' Grafico Errore vs Iterazioni'''

ite_n = np.arange(1,ite_newton+1)  
vect_ite_g1 = np.arange(1,ite_g1+1)  
vect_ite_g2 = np.arange(1,ite_g2+1)  
vect_ite_g3 = np.arange(1,ite_g3+1) 



plt.figure(figsize=(20,15))
plt.plot(ite_n, vecErrore_newton, color = 'blue')
plt.plot(vect_ite_g1, vecErrore_g1, color = 'red')
plt.plot(vect_ite_g2, vecErrore_g2, color = 'green')
plt.plot(vect_ite_g3, vecErrore_g3, color = 'yellow')
plt.title("ITERAZIONI VS ERRORE")
plt.legend(['NEWTON','g(x) = x-f(x)e^(x/2)','g(x)=x-f(x)e^(-x/2)','g(x) = x-f(x)/df(x)']) 
plt.xlabel('Iterazioni')
plt.ylabel('Errore')
plt.show()

