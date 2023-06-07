# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:11:45 2023

@author: giuse
"""
import numpy as np
import matplotlib.pyplot as plt


'''
con lambda più alto i valori della funzione obbiettivo e del gradiente descrescono più velocemente, soprattutto il gradiente 
rispetto alla funzione obiettivo
'''

def next_step(x,f,grad,lamda): 
  alpha=1.1
  alpha_min=10**(-10)
  rho = 0.5
  c1 = 0.25
  p=-grad
  j=0
  jmax=10
  while((f(x + alpha*p,b,lamda)>f(x,b,lamda)+c1*alpha*grad.T@p) and (j<jmax) and (alpha>alpha_min)):
      alpha=rho*alpha
      j=j+1
  if (j==jmax or alpha<=alpha_min):    
      return -1     
  else:
      return alpha  
    



def minimize(f,grad_f,x0,step_fixed,MAXITERATION,ABSOLUTE_STOP, boolean_step_fixed, lamda): 
  x=np.zeros((3,MAXITERATION))  
  norm_grad_list=np.zeros((1,MAXITERATION)) 
  function_eval_list=np.zeros((1,MAXITERATION))    
  k=0
  x_last = np.array([x0[0],x0[1],x0[2]])   
  x[:,k] = x0        
  function_eval_list[:,k]=abs(f(x0,b,lamda))    
  norm_grad_list[:,k]=np.linalg.norm(grad_f(x0, b, lamda))
  
  while (np.linalg.norm(grad_f(x_last, b, lamda))>=ABSOLUTE_STOP and k < MAXITERATION - 1 ):      
      k=k+1
      grad=grad_f(x_last, b, lamda)  
      
      if(boolean_step_fixed): 
          step=step_fixed
      else: 
          step = next_step(x_last, f, grad,lamda)
      if(step==-1):
          print("non converge")
          return
     
      x_last=x_last-(step*grad)
      x[:,k] = x_last
      function_eval_list[:,k]= abs(f(x_last,b,lamda))
      norm_grad_list[:,k]= np.linalg.norm(grad_f(x_last, b, lamda))
  function_eval_list = function_eval_list[:,0:k+1]
  norm_grad_list = norm_grad_list[:,0:k+1]
  x = x[:,0:k+1]
  
  
  return (x_last, norm_grad_list,function_eval_list,k,x)





def f(x,b, lamda):
    res = (x[0]-b[0])**2 + (x[1]-b[1])**2 + (x[2]-b[2])**2 + lamda * x[0]**2 + lamda * x[1]**2 + lamda * x[2]**2
    return res

def grad_f(x, b, lamda):
    return 2*(1+lamda) * x - 2*b

step=0.1
MAXITERATIONS=80 
ABSOLUTE_STOP=1.e-5



lamda1 = 0.2
lamda2 = 0.5
lamda3 = 0.8

n = 3

b = np.ones(n)
x0 = np.array([2,-3,4])  


x_last_1, norm_grad_list_1, function_eval_list_1, k_1, x_1 = minimize(f,grad_f,x0,step,MAXITERATIONS,ABSOLUTE_STOP, False, lamda1)
x_last_2, norm_grad_list_2, function_eval_list_2, k_2, x_2 = minimize(f,grad_f,x0,step,MAXITERATIONS,ABSOLUTE_STOP, False, lamda2)
x_last_3, norm_grad_list_3, function_eval_list_3, k_3, x_3 = minimize(f,grad_f,x0,step,MAXITERATIONS,ABSOLUTE_STOP, False, lamda3)

x_last_1, norm_grad_list_1_fixed, function_eval_list_1_fixed, k_1_fixed, x_1 = minimize(f,grad_f,x0,step,MAXITERATIONS,ABSOLUTE_STOP, True, lamda1)
x_last_2, norm_grad_list_2_fixed, function_eval_list_2_fixed, k_2_fixed, x_2 = minimize(f,grad_f,x0,step,MAXITERATIONS,ABSOLUTE_STOP, True, lamda2)
x_last_3, norm_grad_list_3_fixed, function_eval_list_3_fixed, k_3_fixed, x_3 = minimize(f,grad_f,x0,step,MAXITERATIONS,ABSOLUTE_STOP, True, lamda3)


  
k_plot_1 = np.arange(k_1+1)
norm_grad_list_plot_1=np.reshape(norm_grad_list_1[:,0:k_1+1],k_1+1)
function_eval_list_plot_1 = np.reshape(function_eval_list_1[:,0:k_1+1],k_1+1)

k_plot_1_fixed = np.arange(k_1_fixed+1)
norm_grad_list_plot_1_fixed=np.reshape(norm_grad_list_1_fixed[:,0:k_1_fixed+1],k_1_fixed+1)
function_eval_list_plot_1_fixed = np.reshape(function_eval_list_1_fixed[:,0:k_1_fixed+1],k_1_fixed+1)


k_plot_2 = np.arange(k_2+1)
norm_grad_list_plot_2=np.reshape(norm_grad_list_2[:,0:k_2+1],k_2+1)
function_eval_list_plot_2 = np.reshape(function_eval_list_2[:,0:k_2+1],k_2+1)

k_plot_2_fixed = np.arange(k_2_fixed+1)
norm_grad_list_plot_2_fixed=np.reshape(norm_grad_list_2_fixed[:,0:k_2_fixed+1],k_2_fixed+1)
function_eval_list_plot_2_fixed = np.reshape(function_eval_list_2_fixed[:,0:k_2_fixed+1],k_2_fixed+1)


k_plot_3 = np.arange(k_3+1)
norm_grad_list_plot_3=np.reshape(norm_grad_list_3[:,0:k_3+1],k_3+1)
function_eval_list_plot_3 = np.reshape(function_eval_list_3[:,0:k_3+1],k_3+1)

k_plot_3_fixed = np.arange(k_3_fixed+1)
norm_grad_list_plot_3_fixed=np.reshape(norm_grad_list_3_fixed[:,0:k_3_fixed+1],k_3_fixed+1)
function_eval_list_plot_3_fixed = np.reshape(function_eval_list_3_fixed[:,0:k_3_fixed+1],k_3_fixed+1)



n = 20

plt.figure(figsize=(18,7))

plt.loglog(k_plot_1,norm_grad_list_plot_1, color = 'blue')
plt.loglog(k_plot_2,norm_grad_list_plot_2,color = 'yellow')
plt.loglog(k_plot_3,norm_grad_list_plot_3,color = 'red')
plt.legend(['$\lambda$ = ' + str(lamda1),'$\lambda$ = ' + str(lamda2),'$\lambda$ = ' + str(lamda3)]) 
plt.xlabel("Iterazioni", fontsize = n)
plt.ylabel("Norma Gradiente", fontsize = n)
plt.title('Iterazioni vs Norma Gradiente', fontsize = n)
plt.suptitle("step size variabile",fontsize = n)

plt.show()


plt.figure(figsize=(18,7))

plt.loglog(k_plot_1_fixed,norm_grad_list_plot_1_fixed, color = 'blue')
plt.loglog(k_plot_2_fixed,norm_grad_list_plot_2_fixed,color = 'yellow')
plt.loglog(k_plot_3_fixed,norm_grad_list_plot_3_fixed,color = 'red')
plt.legend(['$\lambda$ = ' + str(lamda1),'$\lambda$ = ' + str(lamda2),'$\lambda$ = ' + str(lamda3)]) 
plt.xlabel("Iterazioni", fontsize = n)
plt.ylabel("Norma Gradiente", fontsize = n)
plt.title('Iterazioni vs Norma Gradiente', fontsize = n)
plt.suptitle("step size fisso",fontsize = n)


plt.show()

plt.figure(figsize=(18,7))
plt.loglog(k_plot_1_fixed,norm_grad_list_plot_1_fixed, color = 'blue')
plt.loglog(k_plot_1,norm_grad_list_plot_1, color = 'red')
plt.legend(['step size fisso','step size variabile']) 
plt.xlabel("Iterazioni", fontsize = n)
plt.ylabel("Norma Gradiente", fontsize = n)
plt.title('Iterazioni vs Norma Gradiente', fontsize = n)
plt.suptitle('$\lambda$ = ' + str(lamda1), fontsize = n)

plt.show()


plt.figure(figsize=(18,7))

plt.plot(k_plot_1,function_eval_list_plot_1, color = 'blue')
plt.plot(k_plot_2,function_eval_list_plot_2,color = 'yellow')
plt.plot(k_plot_3,function_eval_list_plot_3,color = 'red')
plt.legend(['$\lambda$ = ' + str(lamda1),'$\lambda$ = ' + str(lamda2),'$\lambda$ = ' + str(lamda3)]) 
plt.xlabel("Iterazioni", fontsize = n)
plt.ylabel("funzione obiettivo", fontsize = n)
plt.title('Iterazioni vs F(x)', fontsize = n)
plt.suptitle("step size variabile",fontsize = n)

plt.show()


plt.figure(figsize=(18,7))

plt.loglog(k_plot_1_fixed,function_eval_list_plot_1_fixed, color = 'blue')
plt.loglog(k_plot_2_fixed,function_eval_list_plot_2_fixed,color = 'yellow')
plt.loglog(k_plot_3_fixed,function_eval_list_plot_3_fixed,color = 'red')
plt.legend(['$\lambda$ = ' + str(lamda1),'$\lambda$ = ' + str(lamda2),'$\lambda$ = ' + str(lamda3)]) 
plt.xlabel("Iterazioni", fontsize = n)
plt.ylabel("funzione obiettivo", fontsize = n)
plt.title('Iterazioni vs F(x)', fontsize = n)
plt.suptitle("step size fisso",fontsize = n)


plt.show()

plt.figure(figsize=(18,7))
plt.loglog(k_plot_1_fixed,function_eval_list_plot_1_fixed, color = 'blue')
plt.loglog(k_plot_1,function_eval_list_plot_1, color = 'red')
plt.legend(['step size fisso','step size variabile']) 
plt.xlabel("Iterazioni", fontsize = n)
plt.ylabel("funzione obiettivo", fontsize = n)
plt.title('Iterazioni vs F(x)', fontsize = n)
plt.suptitle('$\lambda$ = ' + str(lamda1), fontsize = n)

plt.show()










