import math
import numpy as np
import scipy



""" DIRECT METHODS """

# SOLVE WITH CHOLESKY
def cholesky(A, b):
    L = scipy.linalg.cholesky(A, lower=True)
    y = scipy.linalg.solve(L, b, lower=True)
    x = scipy.linalg.solve(L.T, y, lower=False)
    return x



""" ITERATIVE METHODS """

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



""" NON-LINEAR FUNCTIONS """
  
# function f(x) defined in [a,b]
# xTrue : true result
# returns (xk(calculated solution), iterations, errors(per iteration))
def bisection(a, b, f, xTrue=(), tolx=1.e-7, toly=1.e-16) -> tuple[float, int, np.ndarray]:
	if(f(a) * f(b) > 0):
		print("f(a)*f(b) > 0")
		return (-1, -1, np.zeros(0))
	else:
		a_min = bool(f(a) < f(b))
		k = int(math.ceil( np.log(np.absolute((b-a)/toly)) / np.log(2) ))
		err = np.zeros( (k,1) )
		ak, bk, ck = a, b, 0
		for it in range(k):
			ck = ak + (bk - ak) / 2
			fck = f(ck)
			err[it,0] = np.abs(ck - xTrue)
			if(np.abs(fck) < toly): return (ck, it+1, err[:it+1,0])
			elif((fck > 0 and not(a_min)) or (fck < 0 and a_min)): ak = ck
			else: bk = ck
		return (ck, k, err)


# fcomp : empty
def stopCriteria_absolute(xk, xk_prev, fxk, fcomp=1):
	return (np.abs(fxk), np.abs(xk - xk_prev))
#
def stopCriteria_relative(xk, xk_prev, fxk, fcomp):
	return (np.abs(fxk) / fcomp, np.abs((xk - xk_prev) / xk))
# fcomp : empty
def stopCriteria_absolute_eval(xk, xk_prev, fxk, fcomp=(), toly=1.e-6, tolx=1.-10):
  return np.abs(fxk) <= toly and np.abs(xk - xk_prev) <= tolx
#
def stopCriteria_relative_eval(xk, xk_prev, fxk, fcomp, toly=1.e-6, tolx=1.-10):
	return np.abs(fxk) / fcomp <= toly and np.abs((xk - xk_prev) / xk) <= tolx
# df = D[f] = f'
# xTrue : true solution
# returns (xk(calculated solution), iterations, err(xk,xk-1), err(xk,xTrue))
def newton(f, df, xTrue, maxit, x0=0, stopCriteria=stopCriteria_absolute, tolx=1.-10, toly=1.e-6):
	errk = np.zeros(maxit, dtype=float)
	errk[0] = tolx + 1
	errAbs = np.zeros((maxit,1), dtype=float)
	err_f = toly + 1

	fcomp = f(x0)
	xk = x0
	it = 0
	while(errk[it] > tolx and err_f > toly and it < maxit):
		xk_prev = xk
		xk = xk - f(xk) / df(xk)
		err_f, errk[it] = stopCriteria(xk, xk_prev, f(xk), fcomp=fcomp)
		errAbs[it] = np.abs(xk - xTrue)
		it += 1

	return (xk, it, errk[:it], errAbs[:it])


def successiveApproximations(f, g, maxit, xTrue, x0=0, tolx=1.e-10, toly=1.e-6):
  errk = np.zeros(maxit+1, dtype=np.float64)
  errAbs = np.zeros(maxit+1, dtype=np.float64)
  
  
  i= ...
  err[0]=...
  vecErrore[0] = ...
  x = ...

  while (... ): 
    ...
    
  err = ...  
  vecErrore = ...
  return (x, i, err, vecErrore) 
