
import numpy as np
import matplotlib.pyplot as plt
from numan import (
	iteration,
	matrix,
	solve
)



""" 4.2: SUCCESSIVE APPROXIMATIONS """

'''data'''
f = lambda x: np.exp(x) - x**2
df = lambda x: np.exp(x) - 2*x
a = -1.0
b = 1.0
g =	(
	lambda x: x - f(x) * np.exp(x / 2),
	lambda x: x - f(x) * np.exp(-x / 2),
	lambda x: x - f(x) / df(x)
	)
function_n = len(g)
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
labels_x = tuple(["xTrue"]) + tuple("successive approximations g{}".format(i) for i in range(1,function_n+1))
xTrue = -0.7034674
x_succ, it_succ = [[0. for gi in g] for i in range(2)]
errk, errAbs = [[np.array((1)) for gi in g] for i in range(2)]
for i in range(function_n):
	(x_succ[i], it_succ[i], errk[i], errAbs[i]) = solve.successiveApproximations(f, g[i], xTrue=xTrue, maxit=maxit, x0=x0, tolx=tolx, toly=toly)

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
