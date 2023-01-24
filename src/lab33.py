import sys
sys.path.append("lib")
sys.path.append("../lib")
import os
SCRIPT_PATH="C:/users/danie/cloud-drive/programm/projects/unibo/numerical-analysis/src"
os.chdir(SCRIPT_PATH)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg
from numan import (
	iter,
	matrix,
	methods,
	poly,
	prints
)


""" LLS / SVD """

FIGSIZE = (15,7)
FONTSIZE = 7
STEPS = 300


""" 3 """

functions =	(
			lambda x: x * np.exp(x),
			lambda x: 1 / (1 + 25*x),
			lambda x: np.sin(5*x) + 3*x
			)
domains =	(
			(-1, 1),
			(-1, 1),
			(1, 5)
			)


n = (1,2,3,5,7)
N = 10	#punti noti
x = [np.linspace(D[0], D[1], N) for D in domains]
y = [np.array([f(x1) for x1 in D]) for (f,D) in iter.indexSplit([functions, x])]
x_plot = [np.linspace(D[0], D[1], STEPS) for D in domains]


for i in range(len(functions)):
	prints.approxMulti(x[i], y[i], n)