
import numpy as np
import matplotlib.pyplot as plt
from numan import (
	iteration,
	matrix,
	solve
)
import time



""" 4.3: 3 METHODS """

'''data'''
f = (
    lambda x: x**3 + 4*x*np.cos(x) - 2,
    lambda x: x - x**1/3 - 2
)
df =	(
        lambda x: 3*x**2 + 4*np.cos(x) - 4*x*np.sin(x),
        lambda x: 1 - (1/3)*x**(-2/3)
)
a = (
    0,
    3
)
b = (
    2,
    5
)
g =	(
	lambda x: (2 - x**3) / (4*np.cos(x)),
	lambda x: x**1/3 + 2
)
function_n = len(f)
function_names =	(
					"x**3 + 4*x*cos(x) - 2",
					"x - x**1/3 - 2"
)
methods = ("bisection", "newton", "succ. appr.")
methods_n = len(methods)
markers = ('.', 'o', 'x')

tolx = 1.e-10
toly = 1.e-6
maxit = 100
x0 = tolx
#plots
x_steps = 100
plot_size = (8,3*2)
plot_font_title = 7
plot_legend_size = 7

#x,f(x)
labels_x = tuple(tuple("{method} f{n}".format(method=method, n=i) for method in methods) for i in range(1,function_n+1))
x_appr, it_appr, times = [[[0. for m in methods] for fi in f] for i in range(3)]
errk = [[np.array((1)) for m in methods] for fi in f]
for fi in range(function_n):
    times[fi][0] = time.time()
    (x_appr[fi][0], it_appr[fi][0], errk[fi][0]) = solve.bisection(a[fi], b[fi], f[fi], xTrue=0, tolx=tolx, toly=toly)
    times[fi][0] = time.time() - times[fi][0]
    times[fi][1] = time.time()
    (x_appr[fi][1], it_appr[fi][1], errk[fi][1], errAbs) = solve.newton(f[fi], df[fi], xTrue=0, maxit=maxit, x0=x0, tolx=tolx, toly=toly)
    times[fi][1] = time.time() - times[fi][1]
    times[fi][2] = time.time()
    (x_appr[fi][2], it_appr[fi][2], errk[fi][2], errAbs) = solve.successiveApproximations(f[fi], g[fi], xTrue=0, maxit=maxit, x0=x0, tolx=tolx, toly=toly)
    times[fi][2] = time.time() - times[fi][2]

""" print datas """
# print x, f(x)
for fi in range(function_n):
    for (label, x) in iteration.rearrange_lists([labels_x[fi], x_appr[fi]]):
	    print("{label}:\t\tx={x},\tf(x)={y}".format(label=label, x=x, y=f[fi](x)))
# print iterations
for fi in range(function_n):
	for (label, it) in iteration.rearrange_lists([labels_x[fi], it_appr[fi]]):
		print("{label}\t- iterations={it}".format(label=label, it=it))
# print times
for fi in range(function_n):
	for (label, t) in iteration.rearrange_lists([labels_x[fi], times[fi]]):
		print("{label}\t- time={time}".format(label=label, time=t))

""" plot function """
x_plot = [np.linspace(a[fi], b[fi], x_steps + 1) for fi in range(function_n)]
y_plot = [np.array(list(map(f[fi], x_plot[fi]))) for fi in range(function_n)]
#sol_x_plot = [np.linspace(min(x_appr[fi]),max(x_appr[fi]),methods_n) for fi in range(function_n)]
#sol_y_plot = [np.array(list(map(f[fi], sol_x_plot[fi]))) for fi in range(function_n)]

it_x_plot = [[np.linspace(1, it_appr[fi][mi], int(it_appr[fi][mi])) for mi in range(methods_n)] for fi in range(function_n)]
#draw
for fi in range(function_n):
	plt.figure(figsize=plot_size)

	#function
	plt.subplot(2, 1, 1)
	plt.plot(x_plot[fi], y_plot[fi])
	#plt.plot(sol_x_plot, sol_y_plot, marker='o')
	plt.title(function_names[fi], fontsize=plot_font_title)

	#errk
	plt.subplot(2, 1, 2)
	for (it, err, method, marker) in iteration.rearrange_lists([it_x_plot[fi], errk[fi], methods, markers]):
		plt.plot(it, err, label=method, marker=marker)
	plt.title("err-k", fontsize=plot_font_title)
	plt.legend(fontsize=plot_legend_size)

	plt.show()
