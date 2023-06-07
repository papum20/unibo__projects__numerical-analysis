import sys
sys.path.append("lib")

import numpy as np
import time
from numan import (
	poly,
	prints
)




f	= lambda x: np.exp(x) - x**2
df	= lambda x: np.exp(x) - 2*x
a	= -1.0
b	= 1.0
g	=	(
	lambda x, opt: x - f(x) * np.exp(x / 2),
	lambda x, opt: x - f(x) * np.exp(-x / 2),
	lambda x, opt: x - f(x) / df(x)
)

xTrue	= -0.7034674
tolx	= 1.e-10
toly	= 1.e-6
maxit	= 100
x0		= 0



methods = [
	"bisection",
	"newton",
	"succ. appr. 1",
	"succ. appr. 2",
	"succ. appr. 3"
]
times	= []
res		= []
t		= time.time()
res.append(poly.bisection(a, b, f, xTrue=xTrue, tolx=tolx, toly=toly))
times.append(time.time() - t)
t	= time.time()
res.append(poly.newton(f, df, maxit, xTrue, x0, tolx=tolx, toly=toly))
times.append(time.time() - t)

for gk in g:
	t = time.time()
	res.append(poly.successiveApprox(f, df ,maxit=maxit, xTrue=xTrue, g=gk, x0=x0, tolx=tolx, toly=toly))
	times.append(time.time() - t)



prints.funSolve(a, b, f, xTrue=xTrue, methods=methods, res=res, times=times, xvalues=300, shape=(2,2))