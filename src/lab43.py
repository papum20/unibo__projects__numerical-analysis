import sys
sys.path.append("lib")
sys.path.append("../lib")
import os
SCRIPT_PATH="C:/users/danie/cloud-drive/programm/projects/unibo/numerical-analysis/src"
os.chdir(SCRIPT_PATH)

import numpy as np
import time
from numan import (
	iter,
	poly,
	prints
)


'''data'''
f = (
    lambda x: x**3 + 4*x*np.cos(x) - 2,
    lambda x: x - x**1/3 - 2
)
df =	(
        lambda x: 3*x**2 + 4*np.cos(x) - 4*x*np.sin(x),
        lambda x: 1 - (1/3)*x**(-2/3)
)
a = (
    0,
    3
)
b = (
    2,
    5
)
g =	(
	lambda x, opt: (2 - x**3) / (4*np.cos(x)),
	lambda x, opt: x**1/3 + 2
)

f_names =	(
			"x**3 + 4*x*cos(x) - 2",
			"x - x**1/3 - 2"
)

tolx = 1.e-10
toly = 1.e-6
maxit = 100
x0 = [(ak + bk) / 2. for (ak,bk) in iter.indexSplit([a, b])]
# come xTrue prendo x0


""" solutions """

methods = [
	"bisection",
	"newton",
	"succ. appr. 1",
]

for (fk, dfk, ak, bk, gk, x0k, name) in iter.indexSplit([f, df, a, b, g, x0, f_names]):
	print("\nFUNCTION ", name, "\n")
	times = []
	res = []
	t = time.time()
	res.append(poly.bisection(ak, bk, fk, xTrue=x0k, tolx=tolx, toly=toly))
	times.append(time.time() - t)
	t = time.time()
	res.append(poly.newton(fk, dfk, maxit, xTrue=x0k, x0=x0k, tolx=tolx, toly=toly))
	times.append(time.time() - t)
	# in realtà l'esercizio chiede solo una g per ogni f
	t = time.time()
	res.append(poly.successiveApprox(fk, dfk ,maxit=maxit, xTrue=x0k, g=gk, x0=x0k, tolx=tolx, toly=toly))
	times.append(time.time() - t)

	""" plots """

	prints.funSolve(ak, bk, fk, xTrue=x0k, methods=methods, res=res, times=times, xvalues=300, shape=(2,2))