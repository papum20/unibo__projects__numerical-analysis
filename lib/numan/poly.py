import numpy as np


# Funzione per valutare il polinomio p, in un punto x, dati i coefficienti alpha
# c : container
def evaluate(c:np.ndarray, x):
	c = c.reshape((c.size))
	return sum([c[i] * x**i for i in range(c.size)], 0)