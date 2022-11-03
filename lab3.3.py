import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from numan import (
	matrix as mat,
	polynom as pol
)




##########	3	##########


N = 10
functions =	(
			FunctionPlot(np.linspace(-1, 1, N), lambda x: x * np.exp(x)),
			FunctionPlot(np.linspace(-1, 1, N), lambda x: 1 / (1 + 25*x)),
			FunctionPlot(np.linspace(-1, 5, N), lambda x: np.sin(5*x) + 3*x)
			)
for f in functions:
	f.calculate()
	f.draw()

