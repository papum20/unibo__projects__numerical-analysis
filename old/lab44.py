import numpy as np
import matplotlib.pyplot as plt
from numan import (
	iteration,
	matrix,
	solve
)
import time






''' data '''

f	= lambda x1, x2: 10*(x1-1)**2 + (x2-2)**2
df	= lambda x1, x2: np.array([-20*(x1-1), -4*(x2-2)])
xTrue = np.array([1,2])

step = 0.1
maxit = 1000
stop_df = 1.e-5
mode = 'plot_history'
x0 = np.array((3,-5))


solve.gradient(x0, f, df, xTrue=xTrue, mode=mode, maxit=maxit, stop_d=stop_df, step=step)

... minimize ...


print('iterations=',k)
print('last guess: x=(%f,%f)'%(x[0,k],x[1,k]))


v_x0 = np.linspace(-5,5,500)
v_x1 = ...
x0v,x1v = ...
z = ...
   
'''superficie'''
plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(...)
ax.set_title('Surface plot')
plt.show()

'''contour plots'''
if mode=='plot_history':
   contours = plt.contour(...)
   ...
   ...

'''plots'''

# Iterazioni vs Norma Gradiente
plt.figure()
...
plt.title('Iterazioni vs Norma Gradiente')



#Errore vs Iterazioni
plt.figure()
...
plt.title('Errore vs Iterazioni')



#Iterazioni vs Funzione Obiettivo
plt.figure()
...
plt.title('Iterazioni vs Funzione Obiettivo')











