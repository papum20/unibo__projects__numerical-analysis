

import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt


start=10
finish=100

K_D = np.zeros(finish-start)
Err_D = np.zeros(finish-start)


for n in np.arange(start,finish):
    
    D_val = 9
    D_val_u = -4
    D_val_d = -4
    D = np.eye(n,k=0)*D_val + np.eye(n,k=1)*D_val_u + np.eye(n,k=-1)*D_val_d 
    x_true_d=np.ones((n,1))
    b_d = D@x_true_d
    K_D[n-start] = np.linalg.cond(D, 2)
    L_d = scipy.linalg.cholesky (D, lower=True)
    y_d = np.linalg.solve(L_d,b_d)
    my_x_d = np.linalg.solve(L_d.T,y_d)
    Err_D[n-start] = np.linalg.norm((my_x_d-x_true_d), 2)/np.linalg.norm(x_true_d, 2)
    

n = np.arange(start,finish)  

plt.plot(n,K_D)
plt.title('CONDIZIONAMENTO DI D ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(D)')
plt.show()

plt.plot(n,Err_D)
plt.title('Errore relativo D')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err_D= ||my_x-x||/||x||')
plt.show()