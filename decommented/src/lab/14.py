import sys
sys.path.append("lib")

import numpy as np
import time
from numan import (
	poly,
	prints
)


f = (
    lambda x: x - x**1/3 - 2,
)
df =	(
        lambda x: 1 - (1/3)*x**(-2/3),
)
a = (
    3,
)
b = (
    5,
)
g =	(
	lambda x, opt: x**1/3 + 2,
)

f_names =	(
			"x - x**1/3 - 2",
)

tolx = 1.e-10
toly = 1.e-6
maxit = 100
x0 = [(ak + bk) / 2. for (ak,bk) in zip(a, b)]
# come xTrue prendo x0



methods = [
	"bisection",
	"newton",
	"succ. appr. 1",
]

for (fk, dfk, ak, bk, gk, x0k, name) in zip(f, df, a, b, g, x0, f_names):
	print("\nFUNCTION ", name, "\n")
	times	= []
	res		= []
	t = time.time()
	res.append(poly.bisection(ak, bk, fk, xTrue=x0k, tolx=tolx, toly=toly))
	times.append(time.time() - t)
	t = time.time()
	res.append(poly.newton(fk, dfk, maxit, xTrue=x0k, x0=x0k, tolx=tolx, toly=toly))
	times.append(time.time() - t)
	t = time.time()
	res.append(poly.successiveApprox(fk, dfk ,maxit=maxit, xTrue=x0k, g=gk, x0=x0k, tolx=tolx, toly=toly))
	times.append(time.time() - t)



	prints.funSolve(ak, bk, fk, xTrue=x0k, methods=methods, res=res, times=times, xvalues=300, shape=(2,2))