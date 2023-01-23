"""1. matrici e norme """

import numpy as np

#help(np.linalg) # View source
#help (np.linalg.norm)
#help (np.linalg.cond)

n = 2
A = np.array([[1, 2], [0.499, 1.001]])

print ('Norme di A:')
norm1 = np.linalg.norm(A, ord=1)
norm2 = np.linalg.norm(A, ord=2)
normfro = np.linalg.norm(A, 'fro')
norminf = np.linalg.norm(A, np.inf)

print('Norma1 = ', norm1, '\n')
print('Norma2 = ', norm2, '\n')
print('Normafro = ', normfro, '\n')
print('Norma infinito = ', norminf, '\n')

cond1 = np.linalg.cond(A, p=1)
cond2 = np.linalg.cond(A, p=2)
condfro = np.linalg.cond(A, p='fro')
condinf = np.linalg.cond(A, p=np.inf)

print ('K(A)_1 = ', cond1, '\n')
print ('K(A)_2 = ', cond2, '\n')
print ('K(A)_fro =', condfro, '\n')
print ('K(A)_inf =', condinf, '\n')

x = np.ones((2,1))
b = np.dot(A,x)

btilde = np.array([[3], [1.4985]])
xtilde = np.array([[2., 0.5]]).T

# Verificare che xtilde è soluzione di A xtilde = btilde
# A * xtilde = btilde
print ('A*xtilde = ', np.dot(A, xtilde))

deltax = np.linalg.norm(x -xtilde, ord=2)
deltab = np.linalg.norm(b - btilde, ord=2)

print ('delta x = ', deltax)
print ('delta b = ', deltab)


"""2. fattorizzazione lu"""

import numpy as np
import scipy
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
import scipy.linalg.decomp_lu as LUdec 
# help (LUdec)
# help(scipy.linalg.lu_solve )

# crazione dati e problema test
A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ])
x = np.ones(4)
b = A@x

condA = np.linalg.cond(A, p=2)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')


#help(LUdec.lu_factor)
lu, piv =LUdec.lu_factor(A)

print('lu',lu,'\n')
print('piv',piv,'\n')


# risoluzione di    Ax = b   <--->  PLUx = b 
my_x=scipy.linalg.lu_solve((lu, piv), b)

print('my_x = \n', my_x)
print('norm =', scipy.linalg.norm(x-my_x, 2))




"""3. Choleski"""

import numpy as np
import scipy
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
# help (scipy.linalg.cholesky)
# help (scipy.linalg.solve)

# creazione dati e problema test
A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ], dtype=np.float64)
A = np.matmul(A, np.transpose(A))   #per renderla semidefinita positiva
x = np.ones((4,1))
#x = ...
b = A@x

condA = np.linalg.cond(A, p=2)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

# decomposizione di Choleski
L = scipy.linalg.cholesky(A, lower=True)
print('L:', L, '\n')

print('L.T*L =', L.T@L)
print('err = ', scipy.linalg.norm(A-np.matmul(L, np.transpose(L)), 'fro'))

y = scipy.linalg.solve(L, b)
my_x = scipy.linalg.solve(L.T, y)
print('my_x = ', my_x)
print('norm =', scipy.linalg.norm(x-my_x, 'fro'))



"""4. Choleski con matrice di Hilbert"""

import numpy as np
import scipy
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
# help (scipy.linalg.cholesky)
# help (scipy.linalg.hilbert)

# crazione dati e problema test
n = np.random.randint(2, 16, size=None, dtype=int)
A = scipy.linalg.hilbert(n)
x = np.ones((n, 1))
b = A@x

condA = np.linalg.cond(A, p=2)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

# decomposizione di Choleski
L = scipy.linalg.cholesky(A, lower=True)
print('L:', L, '\n')

print('L.T*L =', L.T@L)
print('err = ', scipy.linalg.norm(A-np.matmul(L, np.transpose(L)), 'fro'))

y = scipy.linalg.solve(L, b)
my_x = scipy.linalg.solve(L.T, y)
print('my_x = \n ', my_x)

print('norm =', scipy.linalg.norm(x-my_x, 'fro'))



"""5. Choleski con matrice di matrice tridiagonale simmetrica e definita positiva """

import numpy as np
import scipy
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
# help (scipy.linalg.cholesky)
# help (np.diag)

# crazione dati e problema test
n = np.random.randint(2, 16)
A = np.diag(np.ones(n) * 9, k=0) + np.diag(np.ones(n-1) * (-4), k=1) + np.diag(np.ones(n-1) * (-4), k=-1)
A = np.matmul(A, np.transpose(A))
x = np.ones((n,1))
b = A@x

condA = np.linalg.cond(A, p=2)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

# decomposizione di Choleski
L = scipy.linalg.cholesky(A)
print('L:', L, '\n')

print('L.T*L =', L.T@L)
print('err = ', scipy.linalg.norm(A-np.matmul(L, np.transpose(L)), 'fro'))

y = scipy.linalg.solve(L, b)
my_x = scipy.linalg.solve(L.T, y)
print('my_x = \n ', my_x)

print('norm =', scipy.linalg.norm(x-my_x, 'fro'))




"""6. plots """

"""
LA MATRICE DI HILBERT È DEFINITA POSITIVA, MA È MOLTO MAL CONDIZIONATA,
PER CUI CON L'AUMENTARE DELLE DIMENSIONI (>=15) PER ERRORI ALGORITMICI/DI ARROTONDAMENTPO
RISULTA NON DEFINITA POSITIVA
"""

import scipy.linalg.decomp_lu as LUdec 
import matplotlib.pyplot as plt

SIZE_MIN = 2
SIZE_MAX = 13
BOUND_MIN = SIZE_MIN
BOUND_MAX = SIZE_MAX + 1
SIZE_MIN_N = 10
SIZE_MAX_N = 100
BOUND_MIN_N = SIZE_MIN_N
BOUND_MAX_N = SIZE_MAX_N + 1
DIAGONAL_MAIN_VALUE = 9
DIAGONAL_SUB_VALUE = -4
DIAGONAL_SUP_VALUE = DIAGONAL_SUB_VALUE

K_H = np.zeros((BOUND_MAX-BOUND_MIN,1))
K_D = np.zeros((BOUND_MAX-BOUND_MIN,1))
K_R = np.zeros((BOUND_MAX_N-BOUND_MIN_N,1))
Err_H = np.zeros((BOUND_MAX-BOUND_MIN,1))
Err_D = np.zeros((BOUND_MAX-BOUND_MIN,1))
Err_R = np.zeros((BOUND_MAX_N-BOUND_MIN_N,1))

for n in np.arange(BOUND_MIN, BOUND_MAX):
    x = np.ones((n,1))
    ## H
    # creazione dati e problema test    
    H = scipy.linalg.hilbert(n)
    bH = H@x
    # numero di condizione 
    K_H[n-BOUND_MIN] = np.linalg.norm(H, ord=2) * np.linalg.norm(H.T, ord=2)
    # fattorizzazione 
    LH = scipy.linalg.cholesky(H)
    yH = scipy.linalg.solve(LH, bH)
    ch_x_H = scipy.linalg.solve(LH.T, yH)
    #luH, pivH = LUdec.lu_factor(H)
    #lu_x_H = scipy.linalg.lu_solve((luH, pivH), bH)
    # errore relativo
    Err_H[n-BOUND_MIN] = np.linalg.norm((x-ch_x_H), ord=2) / np.linalg.norm(x, ord=2)
    #Err_H[n-BOUND_MIN] = np.linalg.norm((x-lu_x_H), p=2) / np.linalg.norm(x, p=2)
    
    ## D
    D = np.diag(np.ones(n) * DIAGONAL_MAIN_VALUE, k=0) + np.diag(np.ones(n-1) * DIAGONAL_SUP_VALUE, k=1) + np.diag(np.ones(n-1) * DIAGONAL_SUP_VALUE, k=-1)
    bD = D@x
    
    K_D[n-BOUND_MIN] = np.linalg.norm(D, ord=2) * np.linalg.norm(D.T, ord=2)
    
    LD = scipy.linalg.cholesky(D)
    yD = scipy.linalg.solve(LD, bD)
    ch_x_D = scipy.linalg.solve(LD.T, yD)

    #luD, pivD = LUdec.lu_factor(D)
    #lu_x_D = scipy.linalg.lu_solve((luD, pivD), bD)
    
    Err_D[n-BOUND_MIN] = np.linalg.norm((x-ch_x_D), ord=2) / np.linalg.norm(x, ord=2)
    #Err_D[n-BOUND_MIN] = np.linalg.norm((x-lu_x_D), p=2) / np.linalg.norm(x, p=2)
    
    
for n in np.arange(BOUND_MIN_N, BOUND_MAX_N):
    x = np.ones((n,1))
    #RAND
    R = np.random.randn(n,n)
    bR = R@x
    
    K_R[n-BOUND_MIN_N] = np.linalg.norm(R, ord=2) * np.linalg.norm(R.T, ord=2)
    
    # LR = scipy.linalg.cholesky(R)
    # yR = scipy.linalg.solve(LR, bR)
    # ch_x_R = scipy.linalg.solve(LR.T, yR)

    luR, pivR = LUdec.lu_factor(R)
    lu_x_R = scipy.linalg.lu_solve((luR, pivR), bR)
    
    #Err_R[n-BOUND_MIN_N] = np.linalg.norm((x-ch_x_R), ord=2) / np.linalg.norm(x, ord=2)
    Err_R[n-BOUND_MIN_N] = np.linalg.norm((x-lu_x_R), ord=2) / np.linalg.norm(x, ord=2)
  
    
    
x = np.arange(BOUND_MIN,BOUND_MAX)

# grafico del numero di condizione vs dim
plt.plot(x, K_H)
plt.title('CONDIZIONAMENTO DI H ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(H)')
plt.show()


# grafico errore in norma 2 in funzione della dimensione del sistema
plt.plot(x, Err_H)
plt.title('Errore relativo')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err_H= ||my_x-x||/||x||')
plt.show()


# grafico del numero di condizione vs dim
plt.plot(x, K_D)
plt.title('CONDIZIONAMENTO DI D ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(D)')
plt.show()


# grafico errore in norma 2 in funzione della dimensione del sistema
plt.plot(x, Err_D)
plt.title('Errore relativo')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err_D= ||my_x-x||/||x||')
plt.show()

xR = np.arange(BOUND_MIN_N,BOUND_MAX_N)

# grafico del numero di condizione vs dim
plt.plot(xR, K_R)
plt.title('CONDIZIONAMENTO DI R ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(R)')
plt.show()


# grafico errore in norma 2 in funzione della dimensione del sistema
plt.plot(xR, Err_R)
plt.title('Errore relativo')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err_R= ||my_x-x||/||x||')
plt.show()



