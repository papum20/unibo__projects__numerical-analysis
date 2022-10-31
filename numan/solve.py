import numpy as np
import scipy



# SOLVE WITH CHOLESKY
def cholesky(A, b):
    L = scipy.linalg.cholesky(A, lower=True)
    y = scipy.linalg.solve(L, b, lower=True)
    x = scipy.linalg.solve(L.T, y, lower=False)
    return x


#tol = tolleranza
def jacobi(A,b,x0,maxit,tol, xTrue):
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




def gaussSeidel(A,b,x0,maxit,tol, xTrue):
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