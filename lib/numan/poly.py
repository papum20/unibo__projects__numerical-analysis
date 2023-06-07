import math
import numpy as np
from typing import Callable
from numan import (
	Constants,
	matrix
)




# Funzione per valutare il polinomio p, in un punto x, dati i coefficienti alpha
# c : container
def evaluate(c:np.ndarray, x):
	c = c.reshape((c.size))
	return sum([c[i] * x**i for i in range(c.size)], 0)


"""
SOLVING METHODS
"""

def bisection(
	a:float,
	b:float,
	f:Callable[[float], float],
	xTrue:float,
	tolx:float=Constants.TOL_X,
	toly:float=Constants.TOL_Y
) -> tuple[float|str, int, float, np.ndarray] :

	k = math.ceil(math.log((b - a) / tolx, 2))		# numero minimo di iterazioni per avere un errore minore di tolx
	err_a = np.zeros( (k) )
	fa, fb = f(a), f(b)
	sign_a, sign_b = (fa >= 0), (fb >= 0)	#true if >= 0, else false

	if sign_a == sign_b and fa != 0 and fb != 0:
		print("Error: f(a) * f(b) > 0")
		return ("Error", -1, -1, err_a)
	
	ak, bk, ck = a, b, 0
	for it in range(1, k):

		if b - a < Constants.FLOAT_MANT_MIN:
			print("Error: l'intervallo è troppo piccolo ")
			return ("Error", it, k, err_a)
		ck = ak + (bk - ak) / 2
		fck = f(ck)
		err_a[it] = np.abs(ck - xTrue)
		if np.abs(fck) < toly:		# se f(c) è molto vicino a 0 
			return (ck, it+1, k, err_a[:it+1])
		elif (fck > 0 and fa > fb) or (fck < 0 and fa < fb):
			ak = ck
		else: bk = ck
	return (ck, k, k, err_a)

      

""" successive approximations """
#
def stopCriteria_abs(xk, xk_prev, fxk, fcomp=1):
	return (np.abs(fxk), np.abs(xk - xk_prev))
#
def stopCriteria_rel(xk, xk_prev, fxk, fcomp):
	return (np.abs(fxk) / fcomp, np.abs((xk - xk_prev) / xk))

def g_newton(
	xk:float,
	fdf:list=[],	#must contain f and df in pos. 0,1
	) -> float:
	f, df = fdf[0], fdf[1]
	return xk - f(xk) / df(xk)


def successiveApprox(
	f:Callable[[float], float],
	df:Callable[[float], float],
	maxit:int,
	xTrue:float,
	g:Callable[[float, list], float],
	x0:float=0,
	stopCriteria:Callable = stopCriteria_abs,
	tolx:float=Constants.TOL_X,
	toly:float=Constants.TOL_Y
) -> tuple[float|str, int, np.ndarray, np.ndarray] :

	err_a		= np.zeros(maxit + 1, dtype=float)
	err_k		= np.zeros(maxit, dtype=float)
	err_a[0]	= matrix.errAbsf(x0, xTrue)
	err_k[0]	= tolx + 1
	err_y		= toly + 1

	fcomp	= f(x0)
	xk		= x0
	it		= 0
	while it == 0 or (it < maxit and err_k[it-1] > tolx and err_y > toly): 
		xk_prev	= xk
		xk		= g(xk, [f, df])
		err_y, err_k[it] = stopCriteria(xk, xk_prev, f(xk), fcomp=fcomp)
		err_a[it+1]	= matrix.errAbsf(xk, xTrue)
		it += 1

	return (xk, it, err_k[:it], err_a[:it])  


''' Newton'''

def newton(
	f:Callable[[float], float],
	df:Callable[[float], float],
	maxit:int,
	xTrue:float,
	x0:float=0,
	stopCriteria:Callable = stopCriteria_abs,
	tolx:float=Constants.TOL_X,
	toly:float=Constants.TOL_Y
) -> tuple[float|str, int, np.ndarray, np.ndarray] :
	return successiveApprox(f, df, maxit, xTrue, g_newton, x0, stopCriteria, tolx, toly)