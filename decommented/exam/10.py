


import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.linalg



def p(alpha, x):
  n=alpha.size
  m=x.size 
  A = np.zeros((m,n))
  for i in range(n):
      A[:, i]=x**i
  return A@alpha


def f(x):
    sol=np.zeros(x.size)
    for i in range(x.size):
        sol[i]= math.sin(5*x[i]) + 3*x[i] 
    return sol




for N in np.arange(10, 60, 20):
    i = N
    x = np.linspace(1, 5, N)
    y = f(x)
    N = x.size
    
    for n in range(1,8,2):
        
        
        A = np.zeros((N, n+1))
        
        for i in range(n+1):
            A[:, i]=x**i
          
        
        
        ''' RISOLUZIONE CON EQUAZIONI NORMALI'''
        ATA = A.T@A
        ATy = A.T@y
        
        L = scipy.linalg.cholesky (ATA, lower=True) 
        tmp = np.linalg.solve(L,ATy)
        
        alpha_normali = np.linalg.solve(L.T,tmp)
        
        ''' RISOLUZIONE CON SVD '''
        
        U, s, Vh = scipy.linalg.svd(A)
        
        alpha_svd = np.zeros(s.shape)
        for i in range(n+1):
            ui=U[:,i]
            vi=Vh[i,:]
            alpha_svd = alpha_svd + (np.dot(ui,y)*vi)/s[i]
          
        
        ''' VISUALIZZAZIONE DEI RISULTATI '''
        
        
        
        y_normali = p(alpha_normali, x)
        y_svd = p(alpha_svd, x)
        
        err1 = np.linalg.norm (y-y_normali, 2) 
        err2 = np.linalg.norm (y-y_svd, 2) 
        print ('Errore di approssimazione con Eq. Normali con grado del polinomio n = ', n, ':',  err1, ' con numero di nodi: ', N)
        print ('Errore di approssimazione con SVD con grado del polinomio n = ',n, ':', err2, ' con numero di nodi: ', N)
        
        plt.figure(figsize=(20, 10))
        
        plt.subplot(1, 2, 1)
        plt.plot(x,y,'o')
        plt.plot(x, y_normali, color = 'blue')
        plt.title('Approssimazione tramite Eq. Normali')
        plt.suptitle(" grado del polinomio = " + str(n) + " usando " + str(N) + " dati ", fontsize=30 )
        
        plt.subplot(1, 2, 2)
        plt.plot(x,y,'o')
        plt.plot(x, y_svd, color = 'red')
        plt.title('Approssimazione tramite SVD')
        
        plt.show()