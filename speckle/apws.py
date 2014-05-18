from __future__ import division
from pylab import *


rc('text', usetex=False)

figure(figsize=(10,6))
axes(frameon=False)
xkcd()

x = linspace(-10, 10)
y = linspace(-10, 10)

y_sin = sin(x)-8
y_1 = sin(2*x) + x

plot(y_sin, x)
plot(y_1, x)

ylim(-10, 10)
xlim(-10, 10)
tick_params(\
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    right='off',
    left='off',
    labelbottom='off', # labels along the bottom edge are off
    labelleft='off')
show()
