# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 18:57:09 2023

@author: giuse
"""
import numpy as np
import matplotlib.pyplot as plt

def next_step(x,f,grad): 
  alpha=1.1
  alpha_min=10**(-10)
  rho = 0.5
  c1 = 0.25
  p=-grad
  j=0
  jmax=10
  while((f(x + alpha*p)>f(x)+c1*alpha*grad.T@p) and (j<jmax) and (alpha>alpha_min)):
      alpha=rho*alpha
      j=j+1
  if (j==jmax or alpha<=alpha_min):    
      return -1    
  else:
      return alpha  
    



def minimize(f,grad_f,x0,x_true,step_fixed,MAXITERATION,ABSOLUTE_STOP, boolean_step_fixed): 
  
  x=np.zeros((2,MAXITERATION)) 
  norm_grad_list=np.zeros((1,MAXITERATION)) 
  function_eval_list=np.zeros((1,MAXITERATION))    
  error_list=np.zeros((1,MAXITERATION))      
  k=0
  x_last = np.array([x0[0],x0[1]])    
  x[:,k] = x0        
  function_eval_list[:,k]=abs(f(x0))   
  error_list[:,k]=np.linalg.norm(np.subtract(x_last,x_true))
  norm_grad_list[:,k]=np.linalg.norm(grad_f(x0))
  
  while (np.linalg.norm(grad_f(x_last))>=ABSOLUTE_STOP and k < MAXITERATION - 1 ):      
      k=k+1
      grad=grad_f(x_last)  
 
      if(boolean_step_fixed): 
          step=step_fixed
      else: 
          step = next_step(x_last, f, grad)
      if(step==-1):
          print("non converge")
          return
      
      x_last=x_last-(step*grad)
      x[:,k] = x_last
      function_eval_list[:,k]= abs(f(x_last))
      error_list[:,k]= np.linalg.norm(np.subtract(x_last,x_true))
      norm_grad_list[:,k]= np.linalg.norm(grad_f(x_last))
  function_eval_list = function_eval_list[:,0:k+1]
  error_list = error_list[:,0:k+1]
  norm_grad_list = norm_grad_list[:,0:k+1]
  x = x[:,0:k+1]
  
  return (x_last, norm_grad_list,function_eval_list, error_list,k,x)

'''creazione del problema'''



def f(x):
  return 10*((x[0]-1)**2)+(x[1]-2)**2

def grad_f(x):
    return np.array([20*x[0]-20,2*x[1]-4])

step=0.1
MAXITERATIONS=80 
ABSOLUTE_STOP=1.e-5

x_true=np.array([1,2])
x0 = np.array([3,-5])

x_last_fixed, norm_grad_list_fixed, function_eval_list_fixed, error_list_fixed, k_fixed, x_fixed = minimize(f,grad_f,x0,x_true,step,MAXITERATIONS,ABSOLUTE_STOP, True)
x_last, norm_grad_list, function_eval_list, error_list, k, x = minimize(f,grad_f,x0,x_true,step,MAXITERATIONS,ABSOLUTE_STOP, False)


v_x0 = np.linspace(-5,5,500)
v_x1 = np.linspace(-5,5,500)
x0v,x1v = np.meshgrid(v_x0, v_x1)
X = np.array([x0v,x1v]) 
z = f([x0v,x1v])

plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x0v,x1v,z,cmap='viridis')
ax.set_title('Surface plot')
plt.show()

    
plt.figure(figsize=(25, 10))
plt.subplot(1, 2, 1)
plt.title("insiemi di livello con step size variabile", fontsize = 30)
contours = plt.contour(x0v, x1v, z, levels=30)
plt.plot(x[0,:],x[1,:],'*-')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.title("insiemi di livello con step size fisso", fontsize = 30)
contours = plt.contour(x0v, x1v, z, levels=30)
plt.plot(x_fixed[0,:],x_fixed[1,:],'*-')
plt.axis('equal')
plt.show()



k_plot = np.arange(k+1)
norm_grad_list_plot=np.reshape(norm_grad_list[:,0:k+1],k+1)
error_list_plot = np.reshape(error_list[:,0:k+1],k+1)
function_eval_list_plot = np.reshape(function_eval_list[:,0:k+1],k+1)



k_plot_fixed = np.arange(k_fixed+1)
norm_grad_list_plot_fixed=np.reshape(norm_grad_list_fixed[:,0:k_fixed+1],k_fixed+1)
error_list_plot_fixed = np.reshape(error_list_fixed[:,0:k_fixed+1],k_fixed+1)
function_eval_list_plot_fixed = np.reshape(function_eval_list_fixed[:,0:k_fixed+1],k_fixed+1)


plt.figure(figsize=(25, 10))
plt.plot(k_plot,norm_grad_list_plot, color = 'blue')
plt.plot(k_plot_fixed,norm_grad_list_plot_fixed, color = 'red')
plt.legend(['step size variabile','step size fisso']) 
plt.xlabel("Iterazioni")
plt.ylabel("Norma Gradiente")
plt.title('Norma Gradiente vs Iterazioni', fontsize= 20)
plt.show()



plt.figure(figsize=(25, 10))
plt.plot(k_plot,error_list_plot, color = 'blue')
plt.plot(k_plot_fixed,error_list_plot_fixed, color = 'red')
plt.legend(['step size variabile','step size fisso']) 
plt.xlabel("Iterazioni")
plt.ylabel("Errore relativo")
plt.title('ERRORE vs Iterazioni', fontsize= 20)
plt.show()


plt.figure(figsize=(25, 10))
plt.plot(k_plot,function_eval_list_plot, color = 'blue')
plt.plot(k_plot_fixed,function_eval_list_plot_fixed, color = 'red')
plt.legend(['step size variabile','step size fisso']) 
plt.xlabel("Iterazioni")
plt.ylabel("Funzione obiettivo")
plt.title('F(x) vs Iterazioni', fontsize= 20)
plt.show()





