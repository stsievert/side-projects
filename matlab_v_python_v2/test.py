
from pylab import *
from time import time
from numba import autojit

# plain for
def plainFor():
    ans = 0
    for i in arange(1e6):
        ans += i
def vecFor():
    x = arange(1e8)
    y = sum(x)

def svdTime():
    n = 524
    x = rand(n,n)
    u,s,v = svd(x)

def cumSumTime():
    x = arange(1e7)
    y = cumsum(x)

def euler():
    a = arange(1,1e3)
    b = arange(1,1e3)
    A, B = meshgrid(a, b)
    c2 = A**2 + B**2;
    C = np.sqrt(c2)
    i = argwhere(A+B+C == 1000)
    ans = A[i[0], i[1]] * B[i[0], i[1]] * C[i[0], i[1]]
    return ans

start = time()
plainFor()
end = time()
forLoop = end - start

# vectorized for
start = time()
vecFor()
end = time()
vecFor = end - start

start = time()
svdTime()
end = time()
svdTime = end - start

start = time()
cumSumTime()
end = time()
cumSumTime = end - start



