import numpy as np
import matplotlib.pyplot as plt

from numan import (
	draw,
	matrix as mat,
	polynom as pol
)
		




##########	1	##########

x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3])
y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5])
n = 5
plot_size = (8, 4)
steps = 300
reg1 = draw.RegressionPlot(x, y, n, plot_size, steps)

reg1.calculate()
reg1.debug()
reg1.draw()


