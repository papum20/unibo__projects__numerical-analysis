import numpy as np
from typing import Callable
from numan import (
	matrix
)
import numan




def gradient(
	f:Callable[[np.ndarray], float],
	df:Callable[[np.ndarray], np.ndarray],
	x0:np.ndarray,
	xTrue:np.ndarray,
	tol_df:float=numan.TOL_DF,
	maxit:int=numan.MAXIT,
	alpha0:float=numan.ALPHA_0
) -> tuple[float|str, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray] : 

	x = np.zeros((maxit+1, x0.size))	#maxit rows, x0.size cols
	x[0] = x0
	fx = np.zeros((maxit+1, 1))
	fx[0] = f(x0)
	norm_df = np.zeros((maxit+1, 1)) 
	norm_df[0] = np.linalg.norm(df(x0), ord=2)
	err_a = np.zeros((maxit+1, 1)) 
	err_a[0] = matrix.errAbs(x0, xTrue)

	it = 0
	while (np.linalg.norm(df(x[it]), ord=2) > tol_df and it < maxit):
		alphak = backtrack(x[it], f, df, alpha0=alpha0)
		if(alphak == -1):
			print("Error")
			return ("Error", x, it, fx[:it+1], norm_df[:it+1], err_a[:it+1])
		it += 1
		x[it] = x[it-1] - alphak * df(x[it-1])
		fx[it] = f(x[it])		
		norm_df[it] = np.linalg.norm(df(x[it]), ord=2)		
		err_a[it] = matrix.errAbs(x[it], xTrue)

	return (float(x[it]), x, it, fx[:it+1], norm_df[:it+1], err_a[:it+1])


def backtrack(
	xk:np.ndarray,
	f:Callable[[np.ndarray], float],
	df:Callable[[np.ndarray], np.ndarray],	# gradient
	pk:np.ndarray=np.zeros((0)),			# direction
	rho:float=numan.RHO,
	c1:float=numan.C1,
	alpha0:float=numan.ALPHA_0,
	maxit:int=numan.MAXIT_ALPHA,
	tol_alpha:float=numan.TOL_ALPHA,
) -> float:
	alpha = alpha0
	# if pk not specified, by default is -gradient
	if pk.size == 0: pk = -df(np.ones(xk.size))
	it = 0
	while not(armijo(xk, f, df, alpha, c1=c1, pk=pk)) and it < maxit and alpha > tol_alpha:
		alpha *= rho
		it += 1
	if it >= maxit or alpha <= tol_alpha:
		print("Error: backtracking not converging")
		return -1
	else:
		return alpha

def armijo(
	xk:np.ndarray,
	f:Callable[[np.ndarray], float],
	df:Callable[[np.ndarray], np.ndarray],
	alpha,
	c1:float=numan.C1,
	pk:np.ndarray=np.zeros((0))
) -> bool:
	return f(xk + alpha*pk) <= f(xk) + c1*alpha * df(xk)@pk