
from __future__ import division
from pylab import *
from scipy.io import loadmat

def find_rho(x, y):
    rho = cov(x, y) / (std(x) * std(y))
    return rho[0,1]
def runningMeanFast(x, N):
    y = np.convolve(x, np.ones((N,))/N)[(N-1):]
    y = y[:len(y)-N]
    return y


seed(42)
N = 1024 * pow(2, 0)

t = linspace(0, 1, num=N)
M = 10
x_stay_noisefree = M*sin(2*pi*t)
x_stay = x_stay_noisefree + randn(N)
y_stay = M*sin(2*pi*t  +  pi/2 )

rho_noavg = find_rho(x_stay_noisefree, y_stay)
print rho_noavg

rhos = []
for MEAN_N in arange(1, N/1-4):
    x = runningMeanFast(x_stay, MEAN_N)
    y = runningMeanFast(y_stay, MEAN_N)

    rho = find_rho(x, y)
    rhos += [rho]

figure()
plot(x_stay, label='noisy input')
plot(x, label='averaged')
plot(y_stay, label='noise free input')
plot(y, label='average')
title('Input signals')
xlabel('$t$')
ylabel('$x(t)')
legend()
savefig('input.png', dpi=300)
show()

figure()
semilogy(rhos)
title('Correlation coefficient')
xlabel('Running average length')
ylabel('Correlation coefficent')
savefig('rhos.png', dpi=300)
show()
