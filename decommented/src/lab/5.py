import sys
sys.path.append("lib")

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec
import time

from numan import (
	matrix,
	methods
)


""" ** METODI ITERATIVI ** """
FIGSIZE		= (15,7)
DPI			= 100
FONTSIZE	= 7
    


print("""\n\n TEMPI \n""")
start, end, step = 10, 100, 10
xaxis = np.arange(start, end, step)

maxit	= 2000
tol		= 1.e-8

tLU	= np.zeros(np.size(xaxis))
tCH	= np.zeros(np.size(xaxis))
tJ	= np.zeros(np.size(xaxis))
tGS	= np.zeros(np.size(xaxis))

for i, n in enumerate(xaxis):

	A		= matrix.tridiagonal(n, 2, 1, 1)
	xTrue	= np.ones((n,1))
	b		= A@xTrue
	x0		= np.zeros((n,1))
	x0[0]	= 1

	t		= time.time()
	lu, piv	= LUdec.lu_factor(A)
	xLU		= scipy.linalg.lu_solve((lu, piv), b)
	tLU[i]	= time.time() - t

	t		= time.time()
	L		= scipy.linalg.cholesky(A)  
	y		= scipy.linalg.solve(L, b)
	xCH		= scipy.linalg.solve(L.T, y)
	tCH[i]	= time.time() - t

	t		= time.time()
	(xJ, kJ, relErrJ, errIterJ)		= methods.jacobi(A,b,x0,maxit,tol,xTrue)
	tJ[i]	= time.time() - t

	t		= time.time()
	(xGS, kGS, relErrGS, errIterGS)	= methods.gaussSeidel(A,b,x0,maxit,tol,xTrue)
	tGS[i]	= time.time() - t

	print(f"times for n={n}:")
	print(f"lu={tLU}, cholesky={tCH[i]}")
	print(f"jacobi={tJ[i]}, gaussSiedel={tGS[i]}")
	print("K(A)=", matrix.condition(A))
	print(f"solutions for n={n}:")
	print("xTrue=", xTrue)
	print("lu=\n", xLU)
	print("cholesky=\n", xCH)
	print("jacobi=\n", xJ)
	print("gaussSiedel=\n", xGS)


plt.rc("font", size=FONTSIZE)
plt.figure(figsize=FIGSIZE)
plt.subplot(1, 2, 1)
plt.plot(xaxis, tLU, label='LU', color='blue', linewidth=1, marker='o')
plt.plot(xaxis, tCH, label='Cholesky', color='green', linewidth=2, marker='.')
plt.plot(xaxis, tJ, label='Jacobi', color='yellow', linewidth=2, marker=',')
plt.plot(xaxis, tGS, label='Gauss Siedel', color='red', linewidth=2, marker='_')
plt.legend(loc='upper right')
plt.xlabel('N')
plt.ylabel('time')
plt.title('Comparison of the times')
plt.suptitle(f'maxit={maxit}')
plt.subplot(1, 2, 2)
plt.semilogy(xaxis, tLU, label='LU', color='blue', linewidth=1, marker='o')
plt.semilogy(xaxis, tCH, label='Cholesky', color='green', linewidth=2, marker='.')
plt.semilogy(xaxis, tJ, label='Jacobi', color='yellow', linewidth=2, marker=',')
plt.semilogy(xaxis, tGS, label='Gauss Siedel', color='red', linewidth=2, marker='_')
plt.legend(loc='upper right')
plt.xlabel('N')
plt.ylabel('time')
plt.title('Comparison of the times')
plt.suptitle(f'maxit={maxit}')
plt.show()



