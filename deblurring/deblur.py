from __future__ import division
from pylab import *
from scipy.signal import fftconvolve

# this assumes that h is known (a big assumption!) -- that we know that 
# that it's an average of so many pixels.



# actually making the image blurred
x = imread('./stars.jpg')
shape = x.shape[0:2]
x = mean(x, axis=2)
x = x/x.max()



N = 100
h = ones(N)
h.shape = (sqrt(N), sqrt(N))

# y will remain the blurred image
y = fftconvolve(x, h, mode='same')
y = y/y.max()



# making our array, flat
x2 = x.flat[:]
h3 = zeros(size(x2))
h3[:size(h.flat[:])] = h.flat[:]
y = y.flat[:]


y2 = ifft(fft(y) / ( 1 + fft(h3)))
y.shape = x2.shape = y2.shape = shape

error = y - x2


figure()
subplot(221)
imshow(x)
title('Original image')

subplot(222)
imshow(y)
title('Blurred image')

subplot(223)
imshow(error)
title('Error, with reconstruction')

subplot(224)
imshow(x2.real)
title('Reconstructed image')

savefig('figure.png')


