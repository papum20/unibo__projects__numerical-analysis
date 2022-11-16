
import numpy as np
import matplotlib.pyplot as plt
from numan import (
	iteration,
	matrix,
	solve
)



""" 4.3: 3 METHODS """

'''data'''
f = (
    lambda x: x**3 + 4*x*np.cos(x) - 2,
    lambda x: x - x**1/3 - 2
    )
df =    (
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
methods = ("bisection", "newton", "successive approx.")
methods_n = len(methods)

tolx = 1.e-10
toly = 1.e-6
maxit = 100
x0 = 0
#plots
x_steps = 100
plot_size = (8,3*2)
plot_font_title = 7
plot_legend_size = 7

#x,f(x)
labels_x = tuple(["xTrue"]) + tuple("{method} g{n}".format(method=method, n=i) for i in range(1,function_n+1) for method in methods)
x_succ, it_succ = [[[0. for m in methods] for fi in f] for i in range(2)]
errk = [[np.array((1)) for m in methods] for fi in f]
for fi in range(function_n):
    (x_succ[fi][0], it_succ[fi][0], errk[fi][0]) = solve.bisection(a[fi], b[fi], f[fi], xTrue=0, tolx=tolx, toly=toly)
    (x_succ[fi][1], it_succ[fi][1], errk[fi][1], errAbs) = solve.newton(f[fi], df[fi], xTrue=0, maxit=maxit, x0=x0, tolx=tolx, toly=toly)
    (x_succ[fi][2], it_succ[fi][2], errk[fi][2], errAbs) = solve.successiveApproximations(f[fi], g[fi], xTrue=0, maxit=maxit, x0=x0, tolx=tolx, toly=toly)

""" print datas """
# print x, f(x)
for (label, x) in iteration.rearrange_lists([labels_x, [xTrue] + x_succ]):
	print("{label}:\t\tx={x},\tf(x)={y}".format(label=label, x=x, y=f(x)))
# print iterations
for (label, it) in iteration.rearrange_lists([labels_x[1:], it_succ]):
	print("{label} - iterations={it}".format(label=label, it=it))

""" plot function """
x_plot = np.linspace(a, b, x_steps + 1)
y_plot = np.array(list(map(f, x_plot)))
sol_x_plot = np.linspace(xTrue,xTrue,1)
sol_y_plot = np.array([0])

it_k_plot = [np.linspace(1, it_succ[i], int(it_succ[i])) for i in range(function_n)]
it_abs_plot = it_k_plot
#draw
plt.figure(figsize=plot_size)

#function
plt.subplot(3, 1, 1)
plt.plot(x_plot, y_plot)
plt.plot(sol_x_plot, sol_y_plot, marker='o')
plt.title("f(x)", fontsize=plot_font_title)

#errk
plt.subplot(3, 1, 2)
for (it, err, n) in iteration.rearrange_lists([it_k_plot, errk, list(range(function_n))]):
	plt.plot(it, err, label="g{}, errk".format(n))
plt.title("errors abs", fontsize=plot_font_title)
plt.legend(fontsize=plot_legend_size)

#errAbs
plt.subplot(3, 1, 3)
for (it, err, n) in iteration.rearrange_lists([it_k_plot, errAbs, list(range(function_n))]):
	plt.plot(it, err, label="g{}, errAbs".format(n))
plt.title("errors k", fontsize=plot_font_title)
plt.legend(fontsize=plot_legend_size)

plt.show()
