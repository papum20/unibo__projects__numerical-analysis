import numpy as np
import scipy

import solve





# POLYNOMIAL REGRESSION COEFFICIENTS
def reg_coeff_Chol(A, y):
    return solve.cholesky(A.T@A, A.T@y)

def pol_reg_coeff_Svd(A, y):
    (U, s, VT) = scipy.linalg.svd(A)
    n = VT.shape[0]
    return VT.T @ np.array([(U.T[:n]@y).T[i]*(s**-1)[i] for i in range(3)])

# Funzione per valutare il polinomio p, in un punto x, dati i coefficienti alpha
def evaluate(c, x):
  return sum([c[i] * x**i for i in range(c.size)])
# Funzione per valutare il polinomio p, nei punti xi del vettore x, dati i coefficienti alpha
def evaluate_multi(c, xv):
    return np.array(list(map(lambda x: evaluate(c, x), xv)))