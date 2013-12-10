from pylab import *
from helper_functions import *

y = read_temp('madison_old.csv')
x = read_temp('minneapolis_old.csv')

figure()
plot(abs(x - y))
title('\\textrm{Temperature difference}')
ylabel('$^\circ$\\textrm{F}')
xlabel('\\textrm{Day}')
show()

Ex = mean(x)
Ey = mean(y)
Ey2 = var(y) + Ey**2
Ex2 = var(x) + Ex**2

Cov_xy = cov(x,y)[0,1]
Exy = Cov_xy + Ex * Ey

A = array([[2*Ey2, 2*Ey], [2*Ex, 2]])
b = array([2*Exy, 2*Ex])
c = solve(A, b)

a = c[0]
b = c[1]

# we know how much minneapolis is offset from madison! Now, predict the next
# year

y_new = read_temp('madison_new.csv')
x_new = read_temp('minneapolis_new.csv')

x_predict = a*y_new + b

error = abs(x_predict - x_new)

linear = linspace(0, 365, num=365)
o = ones(365)

figure()
plot(x_new, 'b.')
plot(y_new, 'g.')
show()

figure()
plot(error, 'b.')
plot(linear, o*mean(error))
show()
