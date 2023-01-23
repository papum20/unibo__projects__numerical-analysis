import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numan import (
	draw,
	matrix as mat,
	polynom as pol
)
		



##########	2	##########

data = np.array(pd.read_csv("HeightVsWeight.csv"))
x = data[:, 0]
y = data[:, 1]
n = 4
plot_size = (8,4)
steps = 300
reg2 = draw.RegressionPlot(x, y, n, plot_size, steps)

reg2.calculate()
print("data.shape: ", data.shape)
reg2.debug()
reg2.draw()