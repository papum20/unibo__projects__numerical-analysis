import numpy as np
import scipy

from . import solve




# Funzione per valutare il polinomio p, in un punto x, dati i coefficienti alpha
def evaluate(c, x):
  return sum([c[i] * x**i for i in range(c.size)])
# Funzione per valutare il polinomio p, nei punti xi del vettore x, dati i coefficienti alpha
def evaluate_multi(c, xv):
    return np.array(list(map(lambda x: evaluate(c, x), xv)))


# POLYNOMIAL REGRESSION COEFFICIENTS
def regression_cholesky(A, y):
    return solve.cholesky(A.T@A, A.T@y)

def regression_svd(A, y):
    (U, s, VT) = scipy.linalg.svd(A)
    n = VT.shape[0]
    return VT.T @ np.array([(U.T[:n]@y).T[i]*(s**-1)[i] for i in range(n)])


# lagrange form applied to xval in a set of values x
def lagrange_form(x, xval):
    return list(np.prod(list(( (xval-xj)/(xi-xj) if xi!=xj else 1) for xj in x)) for xi in x)
# given data x and f(x), returns a list of elements of x interpolated
def interpolation(data_x, f, x):
    data_y = list(map(lambda x: f(x), data_x))
    res = []
    for xval in x:
        L = lagrange_form(x, xval)
        res.append(sum([L[i]*data_y[i] for i in range(len(x))]))
    return res