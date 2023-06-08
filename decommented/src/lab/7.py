import sys
sys.path.append("lib")

import numpy as np
import pandas as pd
from numan import (
	prints
)


print("""\n\n LLS / SVD \n""")

FIGSIZE		= (15,7)
FONTSIZE	= 7
STEPS		= 300



""" 2 """

data = np.array(pd.read_csv("src/lab/HeightVsWeight.csv"))
x = data[:, 0]
y = data[:, 1]

print("shape of data: ", data.shape)
prints.approx(x, y, 5, STEPS)
