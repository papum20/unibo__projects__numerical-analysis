import numpy as np
from typing import Callable
from numan import (
	Constants,
	matrix
)




def gradient(
	f:Callable[[np.ndarray], float],
	df:Callable[[np.ndarray], np.ndarray],
	x0:np.ndarray,
	xTrue:np.ndarray,
	tol_df:float=Constants.TOL_DF,
	maxit:int=Constants.MAXIT,
	alpha0:float=Constants.ALPHA_0,
	alpha_varies:bool=True
) -> tuple[float|str, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray] : 

	x = np.zeros((maxit+1, x0.size))	#maxit rows, x0.size cols
	x[0] = x0
	fx = np.zeros(maxit+1)
	fx[0] = f(x0)
	norm_df = np.zeros(maxit+1) 
	norm_df[0] = np.linalg.norm(df(x0), ord=2)
	err_a = np.zeros(maxit+1) 
	err_a[0] = matrix.errAbs(x0, xTrue)
	alphak = alpha0

	it = 0
	while (np.linalg.norm(df(x[it]), ord=2) > tol_df and it < maxit):
		if alpha_varies: alphak = backtrack(x[it], f, df, alpha0=alpha0)
		if(alphak == -1):
			print("Error")
			return ("Error", x, it, fx[:it+1], norm_df[:it+1], err_a[:it+1])
		it += 1
		x[it] = x[it-1] - alphak * df(x[it-1])
		fx[it] = f(x[it])		
		norm_df[it] = np.linalg.norm(df(x[it]), ord=2)		
		err_a[it] = matrix.errAbs(x[it], xTrue)

	return (x[it], x[:it+1], it, fx[:it+1], norm_df[:it+1], err_a[:it+1])


def backtrack(
	xk:np.ndarray,
	f:Callable[[np.ndarray], float],
	df:Callable[[np.ndarray], np.ndarray],
	pk:np.ndarray=np.zeros((0)),
	rho:float=Constants.RHO,
	c1:float=Constants.C1,
	alpha0:float=Constants.ALPHA_0,
	maxit:int=Constants.MAXIT_ALPHA,
	tol_alpha:float=Constants.TOL_ALPHA,
) -> float:
	alpha = alpha0
	
	if pk.size == 0: pk = -df(xk)
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
	c1:float=Constants.C1,
	pk:np.ndarray=np.zeros((0))
) -> bool:
	return f(xk + alpha*pk) <= f(xk) + c1*alpha * df(xk).T@pk