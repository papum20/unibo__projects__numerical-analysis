import numpy as np
import matplotlib.pyplot as plt
from numan import (
	iteration,
	matrix,
	solve
)
import time



def next_step(x,grad): # backtracking procedure for the choice of the steplength
  alpha=1.1
  rho = 0.5
  c1 = 0.25
  p=-grad
  j=0
  jmax=10
  while 
  ... 
  if  
  ....
    








'''creazione del problema'''


f	= lambda x1, x2: 10*(x1-1)**2 + (x2-2)**2
df	= lambda x1, x2: np.array([-20*(x1-1), -4*(x2-2)])
xTrue=np.array([1,2])

step=0.1
MAXITERATIONS=1000
ABSOLUTE_STOP=1.e-5
mode='plot_history'
x0 = np.array((3,-5))


... minimize ...



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











