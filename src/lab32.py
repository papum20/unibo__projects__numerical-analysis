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



""" 2 """

data = np.array(pd.read_csv("HeightVsWeight.csv"))
x = data[:, 0]
y = data[:, 1]

print("shape of data: ", data.shape)
prints.approx(x, y, 5, STEPS)
