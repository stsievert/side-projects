
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

def time_python():
    start = time()
    plainFor()
    end = time()
    forLoopT = end - start

    # vectorized for
    start = time()
    vecFor()
    end = time()
    vecForT = end - start

    start = time()
    svdTime()
    end = time()
    svdT = end - start

    start = time()
    cumSumTime()
    end = time()
    cumSumT = end - start
    return forLoopT, vecForT, svdT, cumSumT

def plot_bar(numpy, julia, r, matlab):
    numpy_for    = numpy[0];
    numpy_vecfor = numpy[1];
    numpy_svd    = numpy[2];
    numpy_cumsum = numpy[3];
    julia_for    = julia[0];
    julia_vecfor = julia[1];
    julia_svd    = julia[2];
    julia_cumsum = julia[2];
    r_for    = r[0];
    r_vecfor = r[1];
    r_svd    = r[2];
    r_cumsum = r[3];
    matlab_for    = matlab[0]
    matlab_vecfor = matlab[1]
    matlab_svd    = matlab[2]
    matlab_cumsum = matlab[3]

    figure(figsize=(14,14))
    subplot(221)
    bar(0, julia_for,  color='red', label='Julia')
    bar(1, matlab_for, color='yellow', label='Matlab')
    bar(2, numpy_for,  color='blue', label='NumPy')
    bar(3, r_for,      color='green', label='R')
    legend(loc='best')
    ylabel('Time')
    title('For loop')

    subplot(222)
    bar(0, julia_vecfor,  color='red', label='Julia')
    bar(1, matlab_vecfor, color='yellow', label='Matlab')
    bar(2, numpy_vecfor,  color='blue', label='NumPy')
    bar(3, r_vecfor,      color='green', label='R')
    legend(loc='best')
    ylabel('Time')
    title('Vectorized for-loop')

    subplot(223)
    bar(0, julia_svd,  color='red', label='Julia')
    bar(1, matlab_svd, color='yellow', label='Matlab')
    bar(2, numpy_svd,  color='blue', label='NumPy')
    bar(3, r_svd,      color='green', label='R')
    legend(loc='best')
    ylabel('Time')
    title('SVD')


    subplot(224)
    bar(0, julia_cumsum,  color='red', label='Julia')
    bar(1, matlab_cumsum, color='yellow', label='Matlab')
    bar(2, numpy_cumsum,  color='blue', label='NumPy')
    bar(3, r_cumsum,      color='green', label='R')
    legend(loc='best')
    title('Cumulative Sum')
    ylabel('Time')
    show()

