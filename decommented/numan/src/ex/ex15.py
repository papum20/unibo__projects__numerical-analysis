import sys
sys.path.append("lib")

import numpy as np
from numan import (
	iter,
	polyrn,
	prints
)



f = lambda x: 10 * (x[0] - 1)**2 + (x[1] - 2)**2
df = lambda x: np.array([20 * (x[0] - 1), 2 * (x[1] - 2)])
x0 = np.array([0,0])
xTrue = np.array([1,2])
tol_df = 1.e-5
maxit = 1000
alpha0 = 0.1

x, y = [], []

(xFound, xk, it, fx, norm_df, err_a) = polyrn.gradient(f, df, x0=x0, xTrue=xTrue, tol_df=tol_df, maxit=maxit, alpha0=alpha0, alpha_varies=True)
(xFound2, xk2, it2, fx2, norm_df2, err_a2) = polyrn.gradient(f, df, x0=x0, xTrue=xTrue, tol_df=tol_df, maxit=maxit, alpha0=alpha0, alpha_varies=False)

it_range = np.arange(0, it+1, 1)
it_range2 = np.arange(0, it2+1, 1)
x = sum([[it_range, it_range2] for i in range(3)], [])
y = [
	err_a,
	err_a2,
	fx,
	fx2,
	norm_df,
	norm_df2
]
labels = [
	("err abs", "it"),
	("err abs", "it"),
	("f(x)", "it"),
	("f(x)", "it"),
	("||grad(x)||", "it"),
	("||grad(x)||", "it")
]

print(x)
print("xTrue, x found", xTrue, xFound, xFound2)
print("it: ", it, it2)
prints.plot(x, y, labels, shape=(3,4))