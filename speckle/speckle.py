from __future__ import division
from pylab import *
from scipy.signal import convolve2d
from scipy.special import j1, y1

N = 256
r = 0.2

j = linspace(-1,1,num=N)
x, y = meshgrid(j, j)

# pupil
p = zeros((N,N))
i = argwhere(x**2 + y**2 < r**2)
p[i[:,0], i[:,1]] = 1

h = fft2(p)
h = fftshift(h)

r = rand(N, N)
x = exp(1j*2*pi*r)
x *= p

d = N/10 # delta
y = convolve2d(x, h[N/2-d:N/2+d, N/2-d:N/2+d])

figure()
m = N/2
imshow(abs(h[m-d:m+d,m-d:m+d]))
#imshow(abs(h))
show()

figure()
imshow(abs(y))
colorbar()
show()
