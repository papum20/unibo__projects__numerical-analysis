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


""" 1 """

x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3])
y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5])

prints.approx(x, y, 5, STEPS)

