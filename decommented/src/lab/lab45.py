import sys
sys.path.append("lib")

import matplotlib.pyplot as plt
import numpy as np
from numan import (
    polyrn,
    prints
)


f	= lambda x, l: np.linalg.norm(x - 1, ord=2)**2 + l * np.linalg.norm(x, ord=2)**2
df	= lambda x, l: 2 * ((l + 1) * x - 1)
n	= 10
RAND_RANGES = (1, 10, 100)
lambdas = (np.random.random_sample() * i for i in RAND_RANGES)


plotx = []
ploty = []
labels = [
	[
		("it", "fx"),
		("it", "err_abs"),
		("it", "norm_df")
	]
	for i in range(3)
]

for l in lambdas:
	print(f"\n\nlambda={l}")
	fl	= lambda x: f(x, l)
	dfl	= lambda x: df(x, l)
	x0	= np.zeros(n)
	xTrue = np.ones(n) * (1 / (l + 1))
	(xk, x, it, fx, norm_df, err_a) = polyrn.gradient(fl, dfl, x0, xTrue=xTrue)
	plotx.append([np.arange(it + 1) for i in range(3)])
	ploty.append([fx, err_a, norm_df])
	print(f"xTrue={xk}")
	print(f"solution found={xk},\tin {it} iterations")

prints.plot_async(plotx, ploty, labels, shape=(2, 3))
plt.show()