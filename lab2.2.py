""" ** METODI ITERATIVI ** """



import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


"""
VOGLIO UNA FUNZIONE CHE DISEGNI UN GRAFICO PER L'ERRORE RELATIVO (USANDO JACOBI):
    f(k) = |xk-x*|/|x*| (errore relativo; x* = risultato esatto)
LA FUNZIONE DISEGNA ANCHE UN GRAFICO PER L'ERRORE TRA DUE ITERATI SUCCESSIVI:
    |xk-x(k-1)|
"""
#tol = tolleranza
def Jacobi(A,b,x0,maxit,tol, xTrue):
  n=np.size(x0)     
  ite=0                          #nunmero iterazioni
  x = np.copy(x0)
  norma_it=1+tol                #inizializzato in modo che la condizione del while sia vera per almeno la prima iterazione (dove non si ha x(k-1))
  relErr=np.zeros((maxit, 1))   #errore relativo (|xk-x*|/|x*|)
  errIter=np.zeros((maxit, 1))  #errore iterazioni (|xk-x(k-1)|)
  relErr[0]=np.linalg.norm(xTrue-x0)/np.linalg.norm(xTrue)
  while (ite<maxit and norma_it>tol):
    x_old=np.copy(x)
    for i in range(0,n):
      #x[i]=(b[i]-sum([A[i,j]*x_old[j] for j in range(0,i)])-sum([A[i, j]*x_old[j] for j in range(i+1,n)]))/A[i,i]
      x[i] = ( b[i] - np.dot(A[i,0:i],x_old[0:i]) - np.dot(A[i,i+1:n],x_old[i+1:n]) ) / A[i,i]  #formula Jacobi xk
    ite=ite+1
    norma_it = np.linalg.norm(x_old-x)/np.linalg.norm(x_old)    #|xk-x(k-1)|/|xk|
    relErr[ite-1] = np.linalg.norm(xTrue-x)/np.linalg.norm(xTrue)
    errIter[ite-1] = norma_it
  relErr=relErr[:ite]
  errIter=errIter[:ite]  
  return [x, ite, relErr, errIter]




def GaussSeidel(A,b,x0,maxit,tol, xTrue):
    n=np.size(x0)     
    it=0                          #numero iterazioni
    x = np.copy(x0)
    norma_it=1+tol                #inizializzato in modo che la condizione del while sia vera per almeno la prima iterazione (dove non si ha x(k-1))
    relErr=np.zeros((maxit, 1))   #errore relativo (|xk-x*|/|x*|)
    errIter=np.zeros((maxit, 1))  #errore iterazioni (|xk-x(k-1)|)
    relErr[0]=np.linalg.norm(xTrue-x0)/np.linalg.norm(xTrue)
    while (it<maxit and norma_it>tol):
      x_old=np.copy(x)
      for i in range(0,n):
        #x[i]=(b[i]-sum([A[i,j]*x_old[j] for j in range(0,i)])-sum([A[i, j]*x_old[j] for j in range(i+1,n)]))/A[i,i]
        x[i] = ( b[i] - np.dot(A[i,0:i],x[0:i]) - np.dot(A[i,i+1:n],x_old[i+1:n]) ) / A[i,i]  #formula Jacobi xk
      it=it+1
      norma_it = np.linalg.norm(x_old-x)/np.linalg.norm(x_old)    #|xk-x(k-1)|/|xk|
      relErr[it-1] = np.linalg.norm(xTrue-x)/np.linalg.norm(xTrue)
      errIter[it-1] = norma_it
    relErr=relErr[:it]
    errIter=errIter[:it]  
    return [x, it, relErr, errIter]


def tridiagonal(n, d_val, d_val_u, d_val_d):    #d=val(_u/_d): valore su diagonale (up/down)
    #A = np.diag(np.ones(n) * D_val, k=0) + np.diag(np.ones(n-1) * D-val_u, k=1) + np.diag(np.ones(n-1) * D_val_d, k=-1)    
    return np.eye(n,k=0)*d_val + np.eye(n,k=1)*d_val_u + np.eye(n,k=-1)*d_val_d

""" **  matrice tridiagonale nxn ** """
# help(np.diag)
# help (np.eye)
# n=5
# c = np.eye(n)
# s = np.diag(np.ones(n-1)*2,k=1)
# i = ...
# print('\n c:\n',c)
# print('\n s:\n',s)
# print('\n i:\n',i)
# print('\n c+i:\n',c+i+s)

"""
#creazione del problema test
n = 10
A = tridiagonal(n, 2, 1, 1)
xTrue = np.ones((n,1))
b = A@xTrue

print('\n A:\n',A)
print('\n xTrue:\n',xTrue)
print('\n b:\n',b)


#metodi iterativi
x0 = np.zeros((n,1))
x0[0] = 1
maxit = 2000
tol = 1.e-8

(xJacobi, kJacobi, relErrJacobi, errIterJacobi) = Jacobi(A,b,x0,maxit,tol,xTrue) 
(xGS, kGS, relErrGS, errIterGS) = GaussSeidel(A,b,x0,maxit,tol,xTrue) 

print('\nSoluzione calcolata da Jacobi:' )
for i in range(n):
    print('%0.2f' %xJacobi[i])

print('\nSoluzione calcolata da Gauss Seidel:' )
for i in range(n):
    print('%0.2f' %xGS[i])


# CONFRONTI

# Confronto grafico degli errori di Errore Relativo

rangeJabobi = range (0, kJacobi)
rangeGS = range(0, kGS)


plt.figure()
plt.plot(rangeJabobi, relErrJacobi, label='Jacobi', color='blue', linewidth=1, marker='o'  )
plt.plot(rangeGS, relErrGS, label='Gauss Seidel', color = 'red', linewidth=2, marker='.' )
plt.legend(loc='upper right')
plt.xlabel('iterations')
plt.ylabel('Relative Error')
plt.title('Comparison of the different algorithms')
plt.show()
"""



# VERIFICA CON RAGGIO SPETTRALE

N = 100
A = tridiagonal(N, 2, 1, 1)
x0 = np.zeros((N,1))
x0[0] = 1
xTrue = np.ones((N,1))
b = A@xTrue

maxit = 2000
tol = 1.e-8

(xJacobi, kJacobi, relErrJacobi, errIterJacobi) = Jacobi(A,b,x0,maxit,tol,xTrue) 
(xGS, kGS, relErrGS, errIterGS) = GaussSeidel(A,b,x0,maxit,tol,xTrue) 
rangeJacobi = range (0, kJacobi)
rangeGS = range(0, kGS)

# calcolo matrici di iterazione
E = -np.tril(A, k=-1)
F = -np.triu(A, k=1)
D = A + E + F

N_J = E + F
M_J = D
N_GS = F
M_GS = D - E

J = np.linalg.inv(D) @ (E + F)
L1 = np.linalg.inv(D - E) @ F


spectral_radius_J = max(list(map(lambda x: abs(x), np.linalg.eigvals(J))))
spectral_radius_GS = max(list(map(lambda x: abs(x), np.linalg.eigvals(L1))))

print("spectral radius with Jacobi method for N=100: " + str(spectral_radius_J))
print("spectral radius with Gauss-Siedel method for N=100: " + str(spectral_radius_GS))

rangeJabobi = range (0, kJacobi)
rangeGS = range(0, kGS)

plt.figure()
plt.semilogy(rangeJabobi, relErrJacobi, label='Jacobi', color='blue', linewidth=1, marker='o'  )
plt.semilogy(rangeGS, relErrGS, label='Gauss Seidel', color = 'red', linewidth=2, marker='.' )
plt.legend(loc='upper right')
plt.xlabel('iterations')
plt.ylabel('Relative Error')
plt.title('Comparison of the different algorithms with spectral radius')
plt.suptitle('spectral radiuses: Jacobi={}, Gauss-Siedel={}'.format(round(spectral_radius_J,3), round(spectral_radius_GS,3)))
plt.show()



# ERRORE RELATIVO AL VARIARE DEL NUMERO DI ITERAZIONI
N_VALUES = [50, 100]

for N in N_VALUES:
    A = tridiagonal(N, 2, 1, 1)
    x0 = np.zeros((N,1))
    x0[0] = 1
    xTrue = np.ones((N,1))
    b = A@xTrue
    
    maxit = 1000
    tol = 1.e-8
    
    (xJacobi, kJacobi, relErrJacobi, errIterJacobi) = Jacobi(A,b,x0,maxit,tol,xTrue) 
    (xGS, kGS, relErrGS, errIterGS) = GaussSeidel(A,b,x0,maxit,tol,xTrue) 
    rangeJacobi = range (0, kJacobi)
    rangeGS = range(0, kGS)
    
    plt.figure()
    plt.plot(rangeJacobi, relErrJacobi, label='Jacobi', color='blue', linewidth=1, marker='o'  )
    plt.plot(rangeGS, relErrGS, label='Gauss Seidel', color = 'red', linewidth=2, marker='.' )
    plt.legend(loc='upper right')
    plt.xlabel('iterations')
    plt.ylabel('Relative Error')
    plt.title('Comparison of the different algorithms')
    plt.suptitle('N=' + str(N))
    plt.show()


# COMPORTAMENTO AL VARIARE DI N

N = 100
START = 10
STEP = 20
dim = np.arange(START, N, STEP)

#metodi iterativi
maxit = 2000
tol = 1.e-8

ErrRelF_J = np.zeros(np.size(dim))
ErrRelF_GS = np.zeros(np.size(dim))

ite_J = np.zeros(np.size(dim))
ite_GS = np.zeros(np.size(dim))

i = 0

for n in dim:
    
    #creazione del problema test
    A = tridiagonal(n, 2, 1, 1)
    xTrue = np.ones((n,1))
    x0 = np.zeros((n,1))
    x0[0] = 1
    b = A@xTrue
    
    #errore relativo finale
    (xJ, kJ, ErrRel_J, ErrIte_J) = Jacobi(A,b,x0,maxit,tol,xTrue)
    (xGS, kGS, ErrRel_GS, ErrIte_GS) = GaussSeidel(A,b,x0,maxit,tol,xTrue)
    
    #iterazioni
    ErrRelF_J[i], ite_J[i] = ErrRel_J[kJ-1], ErrIte_J[kJ-1]
    ErrRelF_GS[i], ite_GS[i] = ErrRel_GS[kGS-1], ErrIte_GS[kGS-1]

    i = i+1
    

# errore relativo finale dei metodi al variare della dimensione N
plt.figure()
plt.plot(dim, ErrRelF_J, label='Jacobi', color='blue', linewidth=1, marker='o')
plt.plot(dim, ErrRelF_GS, label='Gauss Siedel', color='red', linewidth=2, marker='.')
plt.legend(loc='upper right')
plt.xlabel('size')
plt.ylabel('Relative Error')
plt.title('Comparison of the different algorithms')
plt.show()

#numero di iterazioni di entrambi i metodi al variare di N
plt.figure()
plt.plot(dim,ite_J, label='Jacobi', color='blue', linewidth=1, marker='o')
plt.plot(dim,ite_GS, label='Gauss Siedel', color='red', linewidth=2, marker='.')
plt.legend(loc='upper right')
plt.xlabel('size')
plt.ylabel('last it')
plt.title('Comparison of the different algorithms')
plt.show()



# TEMPI

import numpy as np
import scipy
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec
import time

N = 100
START = 10
STEP = 10
dim = np.arange(START, N, STEP)

maxit = 5000
tol = 1.e-8

tLU = np.zeros(np.size(dim))
tCH = np.zeros(np.size(dim))
tJ = np.zeros(np.size(dim))
tGS = np.zeros(np.size(dim))

i = 0
for n in dim:
    #dati
    A = tridiagonal(n, 2, 1, 1)
    xTrue = np.ones((n,1))
    b = A@xTrue
    #iterativi
    x0 = np.zeros((n,1))
    x0[0] = 1
    
    #errore relativo finale
    t = time.time()
    lu, piv =LUdec.lu_factor(A)
    scipy.linalg.lu_solve((lu, piv), b)
    tLU[i] = time.time() - t
    
    t = time.time()
    L = scipy.linalg.cholesky(A)  
    y = scipy.linalg.solve(L, b)
    scipy.linalg.solve(L.T, y)
    tCH[i] = time.time() - t
    
    t = time.time()
    Jacobi(A,b,x0,maxit,tol,xTrue)
    tJ[i] = time.time() - t
    
    t = time.time()
    GaussSeidel(A,b,x0,maxit,tol,xTrue)
    tGS[i] = time.time() - t

    i = i+1

plt.figure()
plt.semilogy(dim,tLU, label='LU', color='blue', linewidth=1, marker='o')
plt.semilogy(dim,tCH, label='Cholesky', color='green', linewidth=2, marker='.')
plt.semilogy(dim,tJ, label='Jacobi', color='yellow', linewidth=2, marker=',')
plt.semilogy(dim,tGS, label='Gauss Siedel', color='red', linewidth=2, marker='_')
plt.legend(loc='upper right')
plt.xlabel('N')
plt.ylabel('time')
plt.title('Comparison of the times')
plt.suptitle('maxit={}'.format(maxit))
plt.show()



