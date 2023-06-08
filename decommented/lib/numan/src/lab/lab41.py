import sys
sys.path.append("lib")

import numpy as np
from numan import (
	methods,
	poly,
	prints
)


print('''\n\n Bisection \n''')
print("""\n\n Newton \n""")

f		= lambda x: np.exp(x) - x**2
df		= lambda x: np.exp(x) - 2*x
xTrue	= -0.7034674

a		= -1.0
b		= 1.0
tolx	= 1.e-10
toly	= 1.e-6
maxit	= 100
x0		= 0


methods = [
	"bisection",
	"newton"
]
res = [
	poly.bisection(a, b, f, xTrue=xTrue, tolx=tolx, toly=toly),
	poly.newton(f, df, maxit, xTrue, x0, tolx=tolx, toly=toly)
]

""" plots """

prints.funSolve(a, b, f, xTrue=xTrue, methods=methods, res=res, xvalues=300, shape=(2,2))