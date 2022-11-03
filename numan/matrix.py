import numpy as np




# x = [[x0..xm]]
# deg = max degree (=n)
def vandermonde(x, deg):
    return np.array([x**j for j in range(deg+1)]).T


	
def tridiagonal(n, d_val, d_val_u, d_val_d):    #d=val(_u/_d): valore su diagonale (up/down)
    #A = np.diag(np.ones(n) * D_val, k=0) + np.diag(np.ones(n-1) * D-val_u, k=1) + np.diag(np.ones(n-1) * D_val_d, k=-1)    
    return np.eye(n,k=0)*d_val + np.eye(n,k=1)*d_val_u + np.eye(n,k=-1)*d_val_d