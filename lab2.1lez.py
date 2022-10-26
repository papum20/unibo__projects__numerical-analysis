





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
normfro = np.linalg.norm(A, ord='fro')
norminf = np.linalg.norm(A, ord=np.inf)

print('Norma1 = ', norm1, '\n')
print('Norma2 = ', norm2, '\n')
print('Normafro = ', normfro, '\n')
print('Norma infinito = ', norminf, '\n')


cond1 = np.linalg.cond(A, 1)
cond2 = np.linalg.cond(A, 2)
condfro = np.linalg.cond(A, 'fro')
condinf = np.linalg.cond(A, np.inf)

print ('K(A)_1 = ', cond1, '\n')
print ('K(A)_2 = ', cond2, '\n')
print ('K(A)_fro =', condfro, '\n')
print ('K(A)_inf =', condinf, '\n')

x = np.ones((2,1))
b = np.dot(A,x)

btilde = np.array([[3], [1.4985]])
xtilde = np.array([[2, 0.5]]).T

# Verificare che xtilde è soluzione di A xtilde = btilde
# A * xtilde = btilde
print ('A*xtilde = ', np.dot(A, xtilde))

deltax = np.linalg.norm(x - xtilde, ord=2)
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

condA = np.linalg.cond(A, 2)

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

# crazione dati e problema test
A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ], dtype=np.float64)
A = np.matmul(np.transpose(A), A)
x = np.ones((4,1))
#x = ...
b = A@x

condA = np.linalg.cond(A, 2)

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

print('L.T*L =',L.T@L)
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
n = ...
A = scipy.linalg.hilbert ...
x = ...
b = ...

condA = ...

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

# decomposizione di Choleski
L = scipy.linalg.cholesky ...
print('L:', L, '\n')

print('L.T*L =', ...
print('err = ', scipy.linalg.norm(A-np.matmul(L, np.transpose(L)), 'fro'))

y = ...
my_x = ...
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
n = ...
A = np.diag(...) + ...
A = np.matmul(A, np.transpose(A))
x = ...
b = ...

condA = ...

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

# decomposizione di Choleski
L = scipy.linalg.cholesky ...
print('L:', L, '\n')

print('L.T*L =', ...
print('err = ', scipy.linalg.norm(A-np.matmul(L, np.transpose(L)), 'fro'))

y = ...
my_x = ...
print('my_x = \n ', my_x)

print('norm =', scipy.linalg.norm(x-my_x, 'fro'))




"""6. plots """


import scipy.linalg.decomp_lu as LUdec 
import matplotlib.pyplot as plt

K_A = np.zeros((20,1))
Err = np.zeros((20,1))

for n in np.arange(10,30):
    # crazione dati e problema test
    A = ...
    x = ...
    b = ...
    
    # numero di condizione 
    K_A[n-10] = ...
    
    # fattorizzazione 
    lu ,piv = ...
    my_x = ...
    
    # errore relativo
    Err[n-10] = ...
  
x = np.arange(10,30)

# grafico del numero di condizione vs dim
plt.plot(...)
plt.title('CONDIZIONAMENTO DI A ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(A)')
plt.show()


# grafico errore in norma 2 in funzione della dimensione del sistema
plt.plot(...)
plt.title('Errore relativo')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err= ||my_x-x||/||x||')
plt.show()


