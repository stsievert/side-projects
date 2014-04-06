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

# load the data
data = loadmat('WeightData.mat')
weight = data['wt'].flat[:]
protein = data['pro'].flat[:]
fat = data['fat'].flat[:]
carbs = data['carbs'].flat[:]
cals = data['cals'].flat[:]

# going to one array
food_unavg = array([protein, fat, carbs, cals])

# init'ing the holders
days_avg = []
mean_rho = []
weight_unavg = weight.copy()

# find it over a range of averages
for MEAN_N in arange(1,14):
    # averaging weight and food based on MEAN_N
    weight = runningMeanFast(weight_unavg, MEAN_N)

    # average food
    food = []
    for i in arange(4):
        food += [runningMeanFast(food_unavg[i,:], MEAN_N)]
    food = asarray(food)

    # finding the cross correlation
    correlation = []
    for element in food:
        correlation += [corrcoef(element, y=weight)[0,1]]
    correlation = asarray(correlation)

    # adding them to the plot
    days_avg += [MEAN_N]
    mean_rho += [mean(correlation)]

figure()
plot(weight)
plot(weight_unavg)
show()

figure()
plot(food[0])
plot(food_unavg[0])
show()

figure()
plot(days_avg, mean_rho, marker='o')
show()
