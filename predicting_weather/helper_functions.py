from pandas import read_csv
from pylab import *

def read_temp(filename):
    data = read_csv(filename)
    temp = data['Mean TemperatureF']
    temp = asarray(temp)
    return temp







