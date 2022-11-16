import numpy as np
import matplotlib.pyplot as plt
from numan import (
	iteration,
	matrix,
	solve
)



""" 4.1: BISECTION, NEWTON """

""" data """
f = lambda x: np.exp(x) - x**2
df = lambda x: np.exp(x) - 2*x
a=-1.0
b=1.0
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
labels_x = ("xTrue", "bisection", "newton")
xTrue = -0.7034674
(x_bisect, it_bisect, err_bisect) = solve.bisection(a, b, f, xTrue=xTrue, tolx=tolx, toly=toly)
(x_newton, it_newton, errk_newton, errAbs_newton) = solve.newton(f, df, xTrue=xTrue, maxit=maxit, x0=x0)

""" print datas """
# print x, f(x)
for (label, x) in iteration.rearrange_lists([labels_x, [xTrue, x_bisect, x_newton]]):
	print("{label}:\t\tx={x},\tf(x)={y}".format(label=label, x=x, y=f(x)))
# print iterations
for (label, it) in iteration.rearrange_lists([labels_x[1:], [it_bisect, it_newton]]):
	print("{label} - iterations={it}".format(label=label, it=it))

""" plot function """
x_plot = np.linspace(a, b, x_steps + 1)
y_plot = np.array(list(map(f, x_plot)))

it_bisect_plot = np.linspace(1, it_bisect, it_bisect)
it_newtonk_plot = np.linspace(2, it_newton, it_newton-1)
it_newton_plot = np.linspace(1, it_newton, it_newton)
#draw
plt.figure(figsize=plot_size)

plt.subplot(2, 1, 1)
plt.plot(x_plot, y_plot)
plt.title("f(x)", fontsize=plot_font_title)

plt.subplot(2, 1, 2)
plt.plot(it_bisect_plot, err_bisect, label="bisection")
plt.plot(it_newtonk_plot, errk_newton[1:], label="newton, errk")
plt.plot(it_newton_plot, errAbs_newton, label="newton, errAbs")
plt.title("errors", fontsize=plot_font_title)
plt.legend(fontsize=plot_legend_size)

plt.show()
