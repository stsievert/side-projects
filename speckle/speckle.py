from __future__ import division
from pylab import *
from scipy.signal import convolve2d
from scipy.special import j1, y1

N = 256 # how many points?
r = 0.3 # radius

j = linspace(-1,1,num=N)
x, y = meshgrid(j, j) # grid

# pupil
p = zeros((N,N))
i = argwhere(x**2 + y**2 < r**2)
p[i[:,0], i[:,1]] = 1

h = fft2(p) # since H(fx) = P(x)
h = fftshift(h)

x = exp(1j*2*pi*rand(N,N)) # a bunch of random phases
x *= p # only within the pupil

d = N/15 # delta since our eyes aren't infinitely big
y = convolve2d(x, h[N/2-d:N/2+d, N/2-d:N/2+d])

figure(figsize=(5,5))
m = N/2
imshow(abs(h[m-d:m+d,m-d:m+d]), interpolation='nearest')
tick_params(\
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    right='off',
    left='off',
    labelbottom='off', # labels along the bottom edge are off
    labelleft='off')
title('\\textrm{Impulse Response}')
savefig('impulse_respone.png', dpi=300, bbox_inches='tight', pad_inches=0)
#imshow(abs(h))
show()

figure(figsize=(5,5))

# abs since our eyes detect intensity of pow(h, 2)
imshow(abs(y), interpolation='nearest')

title('\\textrm{Laser on rough surface}')
tick_params(\
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    right='off',
    left='off',
    labelbottom='off', # labels along the bottom edge are off
    labelleft='off')
savefig('speckle.png', dpi=300, bbox_inches='tight', pad_inches=0)
show()
