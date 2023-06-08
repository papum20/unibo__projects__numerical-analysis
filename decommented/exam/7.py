
import pandas as pd
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



data = pd.read_csv('HeightVsWeight.csv')
data = np.array(data)



x = data[:, 0]
y = data[:, 1]


for n in range(1,8):
    
    N = x.size
    
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
    
    
    x_plot = np.linspace(10,80, num=300)
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