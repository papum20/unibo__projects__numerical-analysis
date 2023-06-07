import sys
sys.path.append("lib")

import numpy as np
from numan import (
	polyrn,
	prints
)




f	= lambda x: 10*(x[0]-1)**2 + (x[1]-2)**2
df	= lambda x: np.array([20 * (x[0]-1), 2 * (x[1]-2)])

xTrue	= np.array([1,2])
x0		= np.array((3,-5))
maxit	= 100
tol_df	= 1.e-5


(xk, x, it, fx, norm_df, err_a) = polyrn.gradient(f, df, x0, xTrue, tol_df=tol_df, maxit=maxit, alpha0=0.1, alpha_varies=False)
#Funzione Obiettivo / iterazioni
# Norma Gradiente / iterazioni
#Errore vs Iterazioni
yplot	= [fx, norm_df, err_a]
xplot	= [np.arange(0, it+1, 1) for _ in yplot]
labels	= [
	("iterations", "f(x)"),
	("iterations", "df(x)"),
	("iterations", "err abs")
]

print("iterations: ", it)
print("xTrue, f(xTrue), df(xTrue): ", xTrue, f(xTrue), df(xTrue))
print("x found, f(x), df(x): ", xk, f(xk), df(xk))

prints.optim(f, xplot, yplot, x, labels)
