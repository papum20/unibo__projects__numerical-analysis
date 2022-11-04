import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


# x = [[x0..xm]]
# deg = max degree (=n)
def vandermonde(x, deg):
    return np.array([x**j for j in range(deg+1)]).T

# SOLVE WITH CHOLESKY
def chol_solve(A, b):
    L = scipy.linalg.cholesky(A, lower=True)
    y = scipy.linalg.solve(L, b, lower=True)
    x = scipy.linalg.solve(L.T, y, lower=False)
    return x

# POLYNOMIAL REGRESSION COEFFICIENTS
def pol_reg_coeff_Chol(A, y):
    return chol_solve(A.T@A, A.T@y)

def pol_reg_coeff_Svd(A, y):
    (U, s, VT) = scipy.linalg.svd(A)
    n = VT.shape[0]
    return VT.T @ np.array([(U.T[:n]@y).T[i]*(s**-1)[i] for i in range(3)])

# Funzione per valutare il polinomio p, in un punto x, dati i coefficienti alpha
def p(c, x):
  return sum([c[i] * x**i for i in range(c.size)])
# Funzione per valutare il polinomio p, nei punti xi del vettore x, dati i coefficienti alpha
def p_vec(c, xv):
    return np.array(list(map(lambda x: p(c, x), xv)))




n = 2 # Grado del polinomio approssimante

x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3])
y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5])
A = vandermonde(x, n)

print('Shape of x:', x.shape)
print('Shape of y:', y.shape, "\n")
print("A = \n", A)



''' Risoluzione tramite equazioni normali'''

# calcoliamo la matrice del sistema e il termine noto a parte
alpha_normali = pol_reg_coeff_Chol(A, y)

print("alpha_normali = \n", alpha_normali)


'''Risoluzione tramite SVD'''

(U, s, VT) = scipy.linalg.svd(A)
print('Shape of U:', U.shape)
print('Shape of s:', s.shape)
print('Shape of VT:', VT.shape)

alpha_svd = pol_reg_coeff_Svd(A, y)
print("alpha_normali = \n", alpha_normali)




'''Verifica e confronto delle soluzioni'''

'''CONFRONTO ERRORI SUI DATI '''
y1 = p(alpha_normali, y)
y2 = p(alpha_svd, y)
err1 = np.linalg.norm (y-y1, 2) 
err2 = np.linalg.norm (y-y2, 2)

print ('Errore di approssimazione con Eq. Normali: ', err1)
print ('Errore di approssimazione con SVD: ', err2)


'''CONFRONTO GRAFICO '''

x_plot = np.linspace(1,3,100)
y_normali = p_vec(alpha_normali, x_plot)
y_svd = p_vec(alpha_svd, x_plot)


plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.plot(x_plot, y_normali, label="alpha normali")
plt.title('Approssimazione tramite Eq. Normali')

plt.subplot(1, 2, 2)
plt.plot(x_plot, y_svd, label="alpha svd3")
plt.title('Approssimazione tramite SVD')

plt.show()













