# Notes:
#   the base functions are defined first, and should probably be in their own
#   script. the base functions are i/dwt, i/dwt2, i/dwt2_full -- they're just
#   functions to put the image in a sparse (wavelet) domain.

#   I solely declared them early on to call them later, in IST and ISTreal.
#   Here, I use IST to time in the notebook (in this same folder) and ISTreal
#   as my callable function (you can change iterations/sampling rate/cut off
#   value).

# Some notes on notation and variable names, in ISTreal:
#   cut:    this is the cut off value that is done for iterative soft
#           thresholding. everything below this value is set to 0
#   I:      Our original image/signal.
#   rp:     Our random permuatation of indicies, to ensure random sampling. If we
#           had [1,2,3,4], rp would be [3,1,2,4]
#   tn1:    Stands for t_{n+1} (explained elsewhere, I'm sure)
#   xold1:  Stands for xold_{n-1}
#   ys:     Our sampled measurements
#   y:      Our measurements
#   p:      Our sampling rate

# What IST actually does:
#   The equation for IST is 



from pylab import *
import pywt
import random
from numba import autojit

# THIS CODE WORKS. DON"T CHANGE ANY OF THE (I)DWT FUNCTIONS
@autojit
def dwt(x):
    y = zeros_like(x)
    l = len(x)/2
    i = arange(l)
    y[i] = x[2*i] + x[2*i + 1]
    y[i+l] = x[2*i] - x[2*i + 1]
    y = y / sqrt(2)
    return y

@autojit
def dwt2(x):
    y = x.copy()
    w = y.shape[0]
    i = arange(w)
    y = dwt(y[i])
    y = rot90(y, 3)
    y = dwt(y[i])
    y = fliplr(y).T
    return y

@autojit
def dwt2_order(s, order):
    # order means how many places width is shifted over: the bottom of the
    # "approx" image
    x = s*1.0
    width  = len(x[0,:])
    #height = len(x[:,0])
    for k in range(0, order):
        # do it on each row and column
        y = x[0:width>>k, 0:width>>k]
        y = dwt2(y)
        x[0:width>>k, 0:width>>k] = y
    return x

@autojit
def dwt2_full(x):
    order = int(log2(len(x)))
    return dwt2_order(x, order)

@autojit
def idwt(x):
    l = len(x)/2
    y = zeros_like(x)
    i = arange(l)
    y[2*i] = x[i] + x[i+l]
    y[2*i+1] = x[i] - x[i+l]
    y = y / sqrt(2)
    return y

@autojit
def idwt2(x):
    y = x.copy()
    w = y.shape[0]
    i = arange(w)
    y = idwt(y[i])
    y = rot90(y, 3)
    y = idwt(y[i])
    y = fliplr(y).T
    y = np.round(y)
    return y

@autojit
def idwt2_order(x, order):
    """ assumes x is 2D"""
    x = np.asarray(x)
    x = 1.0*x
    w = x.shape[0]
    for i in range(order, 0, -1):
        y = x[0:w>>i-1, 0:w>>i-1]
        y = idwt2(y)
        x[0:w>>i-1, 0:w>>i-1] = y
    return x

@autojit
def idwt2_full(x):
    order = int(log2(len(x)))
    return idwt2_order(x, order)

# we're done declaring the whole haar wavelet stack. the above declarations are
# all correct

def IST():
    w = 256;
    I = arange(w*w).reshape(w, w)
    #I = imread('/Users/scott/Desktop/not-used-frequently/pictures_for_project/len_std.jpg')
    #I = mean(I, axis=2)
    sz = I.shape
    n = sz[0] * sz[1]
    p = 0.5

    rp = arange(n)
    random.shuffle(rp) # rp is random now
    upper = size(rp) * p
    its = 100
    l = 6; 
    y = I.flat[rp[1:upper]] # the samples

    ys = zeros(sz);
    ys.flat[rp[1:upper]] = y;

    xold = zeros(sz);
    xold1 = zeros(sz);
    tn = 1;
    for i in arange(its):
        tn1 = (1 + sqrt(1 + 4*tn*tn))/2;
        xold = xold + (tn-1)/tn1 * (xold - xold1)
        
        t1 = idwt2_full(xold);
        temp = t1.flat[rp[1:upper]];
        temp2 = y - temp;

        temp3 = zeros(sz);
        temp3.flat[rp[1:upper]] = temp2;
        temp3 = dwt2_full(temp3);

        temp4 = xold + temp3;
        xold = temp4;

        j = abs(xold) < l
        xold[j] = 0
        j = abs(xold) > l
        xold[j] = xold[j] - sign(xold[j])*l
      
        
        xold1 = xold
        xold = xold
        tn = tn1


def ISTreal(I, its=100, p=0.5, cut=6, draw=False):
    sz = I.shape
    n = sz[0] * sz[1]
    #p = 0.5

    rp = arange(n)
    random.seed(42)
    random.shuffle(rp) # rp is random now
    upper = size(rp) * p
    #its = 100
    #l = 6; 
    y = I.flat[rp[1:upper]] # the samples

    ys = zeros(sz);
    ys.flat[rp[1:upper]] = y;

    xold = zeros(sz);
    xold1 = zeros(sz);
    tn = 1;
    if draw: ion()
    for i in arange(its):
        tn1 = (1 + sqrt(1 + 4*tn*tn))/2;
        xold = xold + (tn-1)/tn1 * (xold - xold1)
        
        t1 = idwt2_full(xold);
        temp = t1.flat[rp[1:upper]];
        temp2 = y - temp;

        temp3 = zeros(sz);
        temp3.flat[rp[1:upper]] = temp2;
        temp3 = dwt2_full(temp3);

        temp4 = xold + temp3;
        xold = temp4;


        j = abs(xold) < cut
        xold[j] = 0
        j = abs(xold) > cut
        xold[j] = xold[j] - sign(xold[j])*cut
      
        
        xold1 = xold
        xold = xold
        tn = tn1
        
        if draw:
            imshow(idwt2_full(xold), cmap='gray')
            axis('off')
            title(str(i))
            draw()

    return xold, ys


I = imread('./lenna.jpg')
I = mean(I, axis=2); 

#I = idwt2_full(x)
#figure()
#axis('off')
#imshow(I, cmap='gray')
#savefig('lenna-l=%d.png' % i)
#show()






