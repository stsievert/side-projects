

from __future__ import division
from pylab import *
import time

# for 1D
def init_draw_now(x, y):
    ion()
    line, = plot(x, y)
    return line
def draw_now(fig, x, y):
    fig.set_data(x, y)
    draw()

#x = linspace(0, 2*pi)
#y = sin(2*pi*x)

#fig = init_draw_now(x, y)
#for i in linspace(0,2*pi):
    #y = sin(2*pi*x*0.1*i)
    #draw_now(fig, x, y)

# for 2D

N = 16
first_image = arange(N*N).reshape(N,N)
myobj = imshow(first_image)
for i in arange(N*N):
    first_image.flat[i] = 0
    myobj.set_data(first_image)
    draw()
#import numpy as np
#from mayavi import mlab
#x, y = np.mgrid[0:3:1,0:3:1]
#z = x**2 + y**2
#s = mlab.imshow(z)

#for i in range(10):
    #s.mlab_source.scalars = np.asarray(x*0.1*(i+1), 'd')
