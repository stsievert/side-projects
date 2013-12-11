
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


