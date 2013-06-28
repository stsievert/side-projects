from pylab import *
import pywt
import random

# THIS CODE WORKS. DON"T CHANGE ANY OF THE (I)DWT FUNCTIONS

def dwt(x):
    y = zeros(len(x))*1.0
    l = len(x)/2
    for i in arange(len(x)/2):
        y[i]     = x[2*i] + x[2*i+1]
        y[i + l] = x[2*i] - x[2*i+1]
    y = y / sqrt(2)
    return y


def dwt_full(x):
    """ assumes x is 1d and 2^n"""
    y = x.copy()
    order = int(log(len(x), 2))
    w = len(x)
    t = array([])
    for k in range(0, order):
        t = y[0:w>>k]
        ca, cd = pywt.dwt(t, 'haar')
        y[0:w>>k+1], y[w>>k+1:w>>k] = ca, cd
    return y

def dwt2(y):
    x = np.array(y).copy()
    x = x*1.0
    w,l = x.shape
 #  pdb.set_trace()
    for i in range(w):
        x[i,:] = dwt(x[i,:])
    for i in range(l):
        x[:,i] = dwt(x[:,i])
    return np.round(x)


def dwt2_order(s, order):
    # order means how many places width is shifted over: the bottom of the
    # "approx" image
  # pdb.set_trace()
    x = s*1.0
    width  = len(x[0,:])
    height = len(x[:,0])
    for k in range(0, order):
        # do it on each row and column
        y = x[0:width>>k, 0:width>>k]
        y = dwt2(y)
        x[0:width>>k, 0:width>>k] = y
    del y
    x = x*1.0
    return np.round(x)

def dwt2_full(x):
    order = int(log2(len(x)))
    return dwt2_order(x, order)





def idwt(x):
    l = len(x)/2
    y = zeros(len(x))
    for i in arange(len(x)/2):
        y[2*i]   = x[i] + x[i+l]
        y[2*i+1] = x[i] - x[i+l]
    y = y / sqrt(2)
    return y

def idwt2(y):
    """ assumes x is 2D """
    x = np.array(y)*1.0
    w, l = x.shape
    for i in range(w):
        x[:,i] = idwt(x[:,i])
    for i in range(l):
        x[i,:] = idwt(x[i,:])
    return np.round(x)

def idwt2_order(x, order):
    """ assumes x is 2D"""
    x = np.asarray(x)
    x = 1.0*x
    w, l = x.shape
    w, l = int(w), int(l)
    for i in range(order, 0, -1):
        y = x[0:w>>i-1, 0:l>>i-1]
        y = idwt2(y)
        x[0:w>>i-1, 0:l>>i-1] = y
    return x

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


    #subplot(211)
    #imshow(ys, cmap='gray')
    #subplot(212)
    #imshow(idwt2_full(xold), cmap='gray')
    #show()
        

