
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

def p7():
    N = int(1e6)
    # corresponding to 0,1,2,3....
    n = ones(N, dtype=bool)
    for p in arange(sqrt(N))+2:
        p = int(p)
        i = (arange(N/p)+2)*p
        n[i[where(i<N)]] = 0

    number = arange(N)
    primes = number[n]
    return primes[10001+1]
def fib():
    N = 60
    x = arange(N)+1
    y = cumprod(x)
    return y[-1]
def euler1():
    n = arange(1,1e7)
    i = argwhere((n%3==0) | (n%5==0))
    return sum(n[i])
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

    start = time()
    euler()
    end = time()
    eulerT = end - start

    start = time()
    p7()
    end = time()
    euler_7T = end - start

    start = time()
    fib()
    end = time()
    euler_fib = end - start

    start = time()
    fib()
    end = time()
    euler1 = end - start
    return forLoopT, vecForT, svdT, cumSumT, eulerT, euler_7T, euler_fib, euler1

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

    savefig('speed_1.png', dpi=300)
    show()

def plot_bar2(matlab_euler, numpy_euler, julia_euler, \
          matlab_euler7, numpy_euler7, julia_euler7, \
          matlab_fib, numpy_fib, julia_fib, \
          matlab_euler1, numpy_euler1, julia_euler1):
    figure(figsize=(10,10))
    subplot(221)
    bar(0, matlab_euler, color='yellow', label='Matlab')
    bar(1, numpy_euler, color='blue', label='NumPy')
    bar(2, julia_euler, color='red', label='Julia')
    legend(loc='lower right')
    title('Project Euler \#9')
    ylabel('Time (s)')

    subplot(222)
    bar(0, matlab_euler7, color='yellow', label='Matlab')
    bar(1, numpy_euler7,  color='blue', label='NumPy')
    bar(2, julia_euler7,  color='red', label='Julia')
    legend(loc='lower right')
    title('Project Euler \#7')
    ylabel('Time (s)')

    subplot(223)
    bar(0, matlab_fib, color='yellow', label='Matlab', bottom=1e-5)
    bar(1, numpy_fib,  color='blue', label='NumPy', bottom=1e-5)
    bar(2, julia_fib,  color='red', label='Julia', bottom=1e-5)
    yscale('log')
    legend(loc='best')
    title('Fibonacci(60)')
    ylabel('Time (s)')

    subplot(224)
    bar(0, matlab_euler1, color='yellow', label='Matlab', bottom=1e-5)
    bar(1, numpy_euler1,  color='blue', label='NumPy', bottom=1e-5)
    bar(2, julia_euler1,  color='red', label='Julia', bottom=1e-5)
    yscale('log')
    legend(loc='best')
    title('Project Euler \# 1')
    ylabel('Time (s)')
    show()
    





