


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


def p(alpha, x):
  n=alpha.size
  m=x.size 
  A = np.zeros((m,n))
  for i in range(n):
      A[:, i]=x**i
  return A@alpha


for n in range(1,8):
    
    x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3])
    y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5])
    
    N = x.size 
    
    A = np.zeros((N, n+1))
    
    for i in range(n+1):
        A[:, i]=x**i
      
        
    ''' Risoluzione tramite equazioni normali'''
    
    
    ATA = A.T@A
    ATy = A.T@y
    
    L = scipy.linalg.cholesky (ATA, lower=True) 
    tmp = np.linalg.solve(L,ATy)
    
    alpha_normali = np.linalg.solve(L.T,tmp)
    
    
    '''Risoluzione tramite SVD'''
    
    
    
    U, s, Vh = scipy.linalg.svd(A) 
    
    
    alpha_svd = np.zeros(s.shape)
    
    for i in range(n+1):
        ui=U[:,i]
        vi=Vh[i,:]
        alpha_svd = alpha_svd + (np.dot(ui,y)*vi)/s[i]       
    
    
    
    '''CONFRONTO ERRORI SUI DATI '''
    
    y1 = p(alpha_normali,x)
    y2 = p(alpha_svd,x)
    
    err1 = np.linalg.norm (y-y1, 2) 
    err2 = np.linalg.norm (y-y2, 2) 
    
    
    '''CONFRONTO GRAFICO '''
    
    x_plot = np.linspace(1,3, num=300)
    
    y_normali = p(alpha_normali, x_plot)
    y_svd = p(alpha_svd, x_plot)
    
    
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    plt.plot(x,y,'o')
    plt.plot(x_plot, y_normali, color = 'blue')
    plt.title('Approssimazione tramite Eq. Normali')
    plt.suptitle(" grado del polinomio = " + str(n), fontsize=30)
    
    plt.subplot(1, 2, 2)
    plt.plot(x,y,'o')
    plt.plot(x_plot, y_svd, color = 'red')
    plt.title('Approssimazione tramite SVD')
    
    plt.show()