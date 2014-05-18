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

d = N/10 # delta since our eyes aren't infinitely big
y = convolve2d(x, h[N/2-d:N/2+d, N/2-d:N/2+d])

figure()
m = N/2
imshow(abs(h[m-d:m+d,m-d:m+d]))
savefig('impulse_respone.png', dpi=300)
#imshow(abs(h))
show()

figure()
# abs since our eyes detect intensity of pow(h, 2)
imshow(abs(y))
savefig('speckle.png', dpi=300)
show()
