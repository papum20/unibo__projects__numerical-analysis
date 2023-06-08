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



print("""\n\n  matrice tridiagonale nxn \n""")
n		= 100
A		= matrix.tridiagonal(n, 2, 1, 1)
xTrue	= np.ones((n,1))
b		= A@xTrue
x0		= np.zeros((n, 1))
x0[0]	= 1

maxit	= 2000
tol		= 1.e-8

(xJ, kJ, relErrJ, errIterJ)		= methods.jacobi(A, b, x0, maxit, tol, xTrue) 
(xGS, kGS, relErrGS, errIterGS)	= methods.gaussSeidel(A, b, x0, maxit, tol, xTrue)
rangeJ	= np.arange(0, kJ + 1)
rangeGS	= np.arange(0, kGS + 1)

T_J, T_GS = methods.matIt_jacobi(A), methods.matIt_gaussSeidel(A)
spectral_radius_J	= matrix.spectralRadius(T_J)
spectral_radius_GS	= matrix.spectralRadius(T_GS)

print('\n A:\n', A)
print('\n xTrue:\n', xTrue)
print('\n b:\n', b)

print('\nSoluzione calcolata da Jacobi:' )
for i in range(n):
	print(xJ[i])

print('\nSoluzione calcolata da Gauss Seidel:' )
for i in range(n):
	print(xGS[i])
	
print('spectral radiuses: ')
print(f'Jacobi={spectral_radius_J}, Gauss-Seidel={spectral_radius_GS}')
print("K(A)=", matrix.condition(A))

plt.figure(figsize=FIGSIZE)
plt.subplot(1, 2, 1)
plt.plot(rangeJ, relErrJ, label='Jacobi', color='blue', linewidth=1, marker='o')
plt.plot(rangeGS, relErrGS, label='Gauss Seidel', color = 'red', linewidth=2, marker='.')
plt.legend(loc='upper right', fontsize=FONTSIZE)
plt.xlabel('iterations', fontsize=FONTSIZE)
plt.ylabel('Relative Error', fontsize=FONTSIZE)
plt.suptitle(f'spectral radiuses: Jacobi={round(spectral_radius_J, 3)}, Gauss-Seidel={round(spectral_radius_GS, 3)}', fontsize=FONTSIZE)
plt.subplot(1, 2, 2)
plt.semilogy(rangeJ, relErrJ, label='Jacobi', color='blue', linewidth=1, marker='o')
plt.semilogy(rangeGS, relErrGS, label='Gauss Seidel', color = 'red', linewidth=2, marker='.')
plt.legend(loc='upper right', fontsize=FONTSIZE)
plt.xlabel('iterations', fontsize=FONTSIZE)
plt.ylabel('Relative Error', fontsize=FONTSIZE)
plt.suptitle(f'spectral radiuses: Jacobi={round(spectral_radius_J, 3)}, Gauss-Seidel={round(spectral_radius_GS, 3)}', fontsize=FONTSIZE)
plt.show()



print("""\n\n COMPORTAMENTO AL VARIARE DI N \n""")

start, end, step = 10, 100, 20
xaxis	= np.arange(start, end, step)
maxit	= 2000
tol		= 1.e-8

err_r_J		= np.zeros(np.size(xaxis))
err_r_GS	= np.zeros(np.size(xaxis))
ite_J		= np.zeros(np.size(xaxis))
ite_GS		= np.zeros(np.size(xaxis))
K_J			= np.zeros(np.size(xaxis))
K_GS		= np.zeros(np.size(xaxis))


for i, n in enumerate(xaxis):
	A		= matrix.tridiagonal(n, 2, 1, 1)
	xTrue	= np.ones((n,1))
	x0		= np.zeros((n,1))
	x0[0]	= 1
	b		= A@xTrue
	(xJ, kJ, ErrRel_J, ErrIte_J)		= methods.jacobi(A,b,x0,maxit,tol,xTrue)
	(xGS, kGS, ErrRel_GS, ErrIte_GS)	= methods.gaussSeidel(A,b,x0,maxit,tol,xTrue)
	
	err_r_J[i], ite_J[i]	= ErrRel_J[kJ-1], ErrIte_J[kJ-1]
	err_r_GS[i], ite_GS[i]	= ErrRel_GS[kGS-1], ErrIte_GS[kGS-1]
	K_J[i], K_GS[i]			= kJ, kGS

	J, GS = methods.matIt_jacobi(A), methods.matIt_gaussSeidel(A)
	spectralRadius_J	= matrix.spectralRadius(J)
	spectralRadius_GS	= matrix.spectralRadius(GS)
	print(f"spectral radiuses for size n={n}:")
	print(f"Jacobi={spectralRadius_J}, Gauss-Siedel={spectralRadius_GS}")
	print("K(A)= ", matrix.condition(A))
   
    

plt.rc("font", size=FONTSIZE)
fig = plt.figure(figsize=FIGSIZE)
plt.subplot(2, 3, 1)
plt.plot(xaxis, err_r_J, label='Jacobi', color='blue', linewidth=1, marker='o')
plt.plot(xaxis, err_r_GS, label='Gauss Siedel', color='red', linewidth=2, marker='.')
plt.title('errore relativo finale dei metodi al variare della dimensione N', fontsize=FONTSIZE)
plt.legend(loc='upper left')
plt.xlabel('size')
plt.ylabel('Relative Error')
plt.subplot(2, 3, 4)
plt.semilogy(xaxis, err_r_J, label='Jacobi', color='blue', linewidth=1, marker='o')
plt.semilogy(xaxis, err_r_GS, label='Gauss Siedel', color='red', linewidth=2, marker='.')
plt.title('errore relativo finale dei metodi al variare della dimensione N (log)', fontsize=FONTSIZE)
plt.legend(loc='upper left')
plt.xlabel('size')
plt.ylabel('Relative Error')

plt.subplot(2, 3, 2)
plt.plot(xaxis, ite_J, label='Jacobi', color='blue', linewidth=1, marker='o')
plt.plot(xaxis, ite_GS, label='Gauss Siedel', color='red', linewidth=2, marker='.')
plt.legend(loc='upper right')
plt.xlabel('size')
plt.ylabel('iteration error')
plt.title('errore tra iterazioni dei metodi al variare della dimensione N', fontsize=FONTSIZE)
plt.subplot(2, 3, 5)
plt.semilogy(xaxis, ite_J, label='Jacobi', color='blue', linewidth=1, marker='o')
plt.semilogy(xaxis, ite_GS, label='Gauss Siedel', color='red', linewidth=2, marker='.')
plt.legend(loc='upper right')
plt.xlabel('size')
plt.ylabel('iteration error')
plt.title('errore tra iterazioni dei metodi al variare della dimensione N (log)', fontsize=FONTSIZE)

plt.subplot(2, 3, 3)
plt.plot(xaxis, K_J, label='Jacobi', color='blue', linewidth=1, marker='o')
plt.plot(xaxis, K_GS, label='Gauss Siedel', color='red', linewidth=2, marker='.')
plt.legend(loc='upper right')
plt.xlabel('size')
plt.ylabel('last it')
plt.title('numero di iterazioni dei metodi al variare della dimensione N', fontsize=FONTSIZE)
plt.subplot(2, 3, 6)
plt.semilogy(xaxis, K_J, label='Jacobi', color='blue', linewidth=1, marker='o')
plt.semilogy(xaxis, K_GS, label='Gauss Siedel', color='red', linewidth=2, marker='.')
plt.legend(loc='upper right')
plt.xlabel('size')
plt.ylabel('last it')
plt.title('numero di iterazioni dei metodi al variare della dimensione N (log)', fontsize=FONTSIZE)
plt.show()



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



