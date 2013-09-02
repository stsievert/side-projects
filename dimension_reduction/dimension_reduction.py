from __future__ import division
from pylab import *
from sklearn import datasets, svm
import numpy

dict = datasets.load_digits()

descr = dict['DESCR']
data = dict['data']
img = dict['images']
tar = dict['target']
target_names = dict['target_names']

def fromSKLearn():
    dict = datasets.load_digits()

    descr = dict['DESCR']
    data = dict['data']
    img = dict['images']
    tar = dict['target']
    target_names = dict['target_names']

    # they're the proper digits
    show = data.reshape(1797, 8, 8)

    # now, we have to figure out each number.
    # x is our data
    # w, the weights
    # so, s = w_1*x_1 ... w_n * x_n
    # y = dot(x^T, s)
    # 

    clf = svm.SVC()#gamma=1e-5)
    clf.fit(data, tar)
    pred = clf.predict(data)

    # it's only wrong 0.1% of the time
    error = abs(pred - tar)


# w_1 = \arg \max_{||w=1||}  Var(x^T  w)

# multiply x by the w's
x = data[0]
w = rand(x.shape[0], x.shape[0])

x = asmatrix(x)
w = asmatrix(w)
a = x * w.T




