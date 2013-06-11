from __future__ import division
import math

from numba import autojit, jit
import pudb as pdb
import itertools
import string
import numpy as np
from PIL import Image
from pylab import *

def factorial(n):
    if n==1:# or n==0:
        return 1
    if n == 0:
        return 0
    else:
        return n* factorial(n-1)

def digits(n):
    return int(math.log10(n) + 1)


def find(n, search):
    x = 0
    # assumes n in a string
    for i in range(0,len(n)):
        if n[i] == str(search):
            x += 1 
    return x

def a34():
    # find all digits where it equals the factorial
    # ie, 145 = 1! + 4! + 5! = 1+ 24 + 120 = 145
    x = 0
    list = []
    for y in range(4, int(1e8)):
        total_fact = 0
        if y % 1e6 == 0:
            print y, x 

        for i in range(0,digits(1.0*y)):
            total_fact += factorial(int(str(y)[i]))

        if y == total_fact:
            list.append(y)
            x += 1
    print x
    print list



def a24():
    # find the millionth lexographic permutation of
    # 0,1,2,3,4,5,6,7,8,9

    # start out with lowest number, increasing the next number
    x = []
    i = 0
    for perm in itertools.permutations([0,1,2,3,4,5,6,7,8,9]):
        i += 1
        x.append(perm)

        if i % (1e6  +1)== 0:
            print perm, i
            print x[int(1e6) - 1]
#   x = itertools.permutations([0,1,2,3,4,5,6,7,8,9])
#   print x[int(1e6+1)]

def convert_char(old):  
    if len(old) != 1:
            return 0
    new = ord(old)
    if 65 <= new <= 90:
       # Upper case letter    
       return new - 64
    elif 97 <= new <= 122:
        # Lower case letter
        return new - 96  # Unrecognized character
    return 0

def a21():
    # find the sum of scores in a file of names. scott = 5th name. scott =  (20 + 3...) * 5
    k = 0
    file = open('names.txt')
    sum = 0

    for name in file.readlines():
        k += 1
        score = 0
        name = name[:-1]
        for j in range(0,len(name)):
            score += convert_char(name[j])
        sum += score * k
    print sum
    return sum


def a73():
    list = []
    #db.set_trace()
    for d in range(1,12000 + 1):
        for n in range(1,d+ 1):
            if 1.0*n/d < 1:
                list += [1.0*n/d]
    list.sort()

    n = list
    x, y = 0, 0
    for i in range(0, len(list)):
        if n[i] == 1/2:
            x += 1
        if n[1] == 1/3:
            x += 1
        if x < 2 and x >= 1:
            y += 1
        if x == 2:
            print x

    print y

def imageFromArray(r,g,b):
    im = np.zeros((len(r), len(r[0]), 3))
    im[:,:,0] = r
    im[:,:,1] = g
    im[:,:,2] = b
    im = Image.fromarray(np.uint8(im))
    return im
def color_pick(n):
    """ pick a color based on the iterations """
    r = 0
    g = n/5
    b =  n


    return (r, g, b)

def mandel(c):
    max_iters = 500
    z = 0
    for n in range(max_iters):
        z = z**2 + c
        if z.real > 2 or z.imag > 2:
            break
    
    # inside = high n, outside = low n
    # want black/blue outside, some red. (0,0,0) = black
    # r = 255...0
    # g = 20...10
    # b = 255-210
    r = 001 + 0.1*n
    g = 255 - 0.5*n
    b = 255 - 0.5*n
    # color in r, g, b
    color = color_pick(n)

    if n == max_iters-1:
       color =  (0,0,0)

    return color

#   colors = np.arange(0,256)
#   return colors[n], z.real2 z.r_end = -1

def mandelbrot(x, y, n):
    """ performs mandelbrot from -x...x and -y...y with n horizontal boxes"""
    m = (y[1] - y[0]) / (x[1] - x[0]) * n
    x, y = np.asarray(x), np.asarray(y)
    xx = np.linspace(x[0], x[1], n)
    yy = np.linspace(y[0], y[1], m)
    yy = yy*1j
    iters = []; color = []
    zi = []; zr = []
   #pdb.set_trace()
    for y in yy:
        for x in xx:
            t = mandel(x + y)
          # iters += [t[0]]
            color += [t]
          # zr    += [t[1]]
          # zi    += [t[2]]
          # color += [[x, y, iters]]
    color = np.asarray(color)
    color = np.reshape(color, (m, n, 3))
    r = color[:,:,0]
    g = color[:,:,1]
    b = color[:,:,2]
    im = imageFromArray(r,g,b)
   #im.show()
    return im

def s73():
    """ 
    makes a sorted set of fractions. n/d for d<= 12,000
    """
    list = []
    limit = 12
   #pdb.set_trace()
    for n in range(1,limit+1):
        for d in range(1,limit+1):
            if n/d < 1/2 and n/d > 1/3:
                list += [n/d]
    list.sort()
    return len(list)

def di(d):
    min = 10000
    minx = -1
    miny = -1
    # find minimal solution for x^2 - d*y^2 == 1 with a given d
    for x in range(2,d**3):
        y = 1.0*math.sqrt(1.0*(x**2 - 1)/d)
        if x + y < min and math.floor(y) == y:
            minx = x
            miny = y
            min = x + y
    return minx, miny

def p66():
    maxx = 0
    x = -1
    for d in range(5,1000+1):
        if math.sqrt(d) % 1.0 != 0:
            x, y = di(d)
            print d, x, y
        if x > maxx:
            maxx = x
    print maxx


def p42():
    file = 'words.txt'
    file = open(file)
    x = 0; sum = 0
    triangle = [ n * (n+1)/2 for n in range(1,1000)]

    for line in file:
        sum = 0
        line = line[:-1]
        print line
        for i in range(0,len(line)):
            sum += string.uppercase.index(line[i])
            sum += 1

        if sum in triangle:
            x += 1
    print x

def p43():
    iters = itertools.permutations([0,1,2,3,4,5,6,7,8,9])
    div   = [2,3,5,7,11,13,17]

    for item in iters:
     #  print item
        test = True
        for i in range(1,8):
            number = 100*item[i+0] + 10*item[i+1] + item[i+2]
            if number % div[i-2] != 0:
                test = False
        
        if test:
            sum += [int(item[i]) for i in range(0,10)]
    print sum


def is_prime(n):
    n = 1.0*n
    prime = True
    for divide in range(2,int(math.sqrt(n) + 1)):
        if n/divide % 1.0 == 0:
            prime = False
            return False
    return True

def p50():
    """ 41 can be written as the sum of 6 consecutive primes.
        what prime below 1,000,000 is the sum of the most consecutive
        primes?"""
    return "see below"

def prime_sum(n, list):
    # tries to find a sum from prime_list
    j = 0; 
    i = n/10
    number = -1
    #db.set_trace()
    for i in range(int(2*n)):
        for j in range(100):
            sum_prime = sum(list[i:j])
            if sum_prime == n and number < j-i:
                number = j-i
    return number

def quadratic(a,b,c):
    return (-b + math.sqrt(b**2 - 4*a*c))/(2*a), (-b - math.sqrt(b**2 - 4*a*c))/(2*a)

def triangle(m):
    """ sees if a number is pentagonal
    
     m == 0.5*n**2 + 0.5*n 
    
    """
    a = 0.5
    b = 0.5
    c = -m
    one, two = quadratic(a,b,c)
    if one % 1.0 == 0:
        return True
    else:
        return False

def pentagonal(m):
    """ pn = n(3n-1)/2
        
        0 == 0.5*3*n^2 - n*0.5 - m
    """
    
    a = 0.5*3
    b = -0.5
    c = -m
    one, two = quadratic(a,b,c)
    if one % 1.0 == 0:
        return True
    else:
        return False
    
def hexagonal(m):
    """ hn = 2*n^2 - n"""
    a = 2
    b = -1
    c = -m
    one, two = quadratic(a,b,c)
    if one % 1.0 == 0:
        return True
    else:
        return False
    
def p45():
    # we're starting at 40755
    for n in range(40756, int(1e5)):
        if pentagonal(n) and triangle(n) and hexagonal(n):
            print n
            break

def p(r):
    n = 0
    for i in range(len(r)):
        n += 1/r[i]
    return 1/n

def infinite_network_of_r(n):
    """ makes an infinite network of resistors...

        a network like this:

       ______________vvvv__vvv_____vvv________
       | +  |             |     |       |   
       |    |             k     k       k   
       | -  |             k     k       k   
       ------             |     |       |       
            |-------------------------------
        
        solution: we want the total current. we know the voltage
        two stems: 1/r_tot = 1/2*r + 1/r

        we're using a network of 1e3 ohm resistors
     """

    r = 1e3
    n = int(n)
    # r_end is the resistance between r_vertical loci
    r_end = r + r
    
    for i in range(n-1):
        r_end = p((r, r_end)) + r
    return r_end
def plot():
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import rc
    rc('text', usetex=True)
    min = 1; max = 10
    x = np.array(range(min, max)) - 1
    y = [infinite_network_of_r(n) for n in range(min, max)]
    plt.plot(x, y)
    plt.title('$\\textrm{Effective resistance of an infinite network}$')
    plt.ylabel('$\\textrm{Effective resistance}$')
    plt.xlabel('$\\textrm{Iterations}$')
    plt.show()

def p71():
    """
        Consider the fraction, n/d, where n and d are positive integers. If nd and
        HCF(n,d)=1, it is called a reduced proper fraction.

        If we list the set of reduced proper fractions for d  8 in ascending order of
        size, we get:

        1/8, 1/7, 1/6, 1/5, 1/4, 2/7, 1/3, 3/8, 2/5, 3/7, 1/2, 4/7, 3/5, 5/8, 2/3, 5/7,
        3/4, 4/5, 5/6, 6/7, 7/8

        It can be seen that 2/5 is the fraction immediately to the left of 3/7.

        By listing the set of reduced proper fractions for d  1,000,000 in ascending
        order of size, find the numerator of the fraction immediately to the left of
        3/7.
    """
    # n/d = 3/7
    # n = 3*d/7
    d = 1e6
    min = (100, 100)
    for d in range(8, int(1e6)):
        for n in range(int(3*d/7 - 5), int(3*d/7 + 5)):
            error = abs( (3/7) - n/d)
            if error < min[0] and error != 0:
                min = (error, n, d)
               #print min
    print min

def find_divisors(n):
    div = np.array([])
    for i in range(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            div = np.append(div, [i])
    for i in range(len(div)):
        div = np.append(div, [n/div[i]])
    return div

def reduced_fraction(n, d):
    """ tests to see if n/d a fraction in the simplest form """
    n, d = 1.0*n, 1.0*d
    if n == 1:
        return True
    
    div = find_divisors(d)

    for i in range(1, len(div)):
        if n % div[i] == 0:
            return False
    return True

    


def p72():
    """
        Consider the fraction, n/d, where n and d are positive integers. If nd and
        HCF(n,d)=1, it is called a reduced proper fraction.

        If we list the set of reduced proper fractions for d  8 in ascending order of
        size, we get:

        1/8, 1/7, 1/6, 1/5, 1/4, 2/7, 1/3, 3/8, 2/5, 3/7, 1/2, 4/7, 3/5, 5/8, 2/3, 5/7,
        3/4, 4/5, 5/6, 6/7, 7/8

        It can be seen that there are 21 elements in this set.

        How many elements would be contained in the set of reduced proper fractions for
        d  1,000,000?
    """
    # use a sieve of sorts. get 1/4, then cross off 2/8, 3/12, 4/16, 5/20....
    n = np.arange(1, 1e2)*1.0
    d = np.arange(1, 1e2)*1.0
    n, d = np.meshgrid(n, d)


    return n, d

# the below two functions are to go from (((1,2),3),4) to [1, 2, 3, 4]
def listit(t):
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t
def flat(mlist):
    result = []
    for i in mlist:
        if type(i) is list:
            result.extend(flat(i))
        else:
            result.append(i)
    return result
def primeFactors(n, p):
    """ Finds p distint prime factors of n """
    # make recursive?
    if p==1:
        return
    if p==2:
        for m in arange(int(sqrt(n)))+2:
            i = n/m
            if i==int(i) and is_prime(m) and is_prime(i):
                return i, m

    if p >= 3:
        # find the prime factor first
        for m in arange(int(sqrt(n))) + 2:
            i = n/m
            if i == int(i) and is_prime(m):
                factor = m
                factors = (primeFactors(i, p-1), factor)
                if factors[0] == None:
                    factors = (primeFactors(i, p), factor)
                return factors


def p47():
    """ The first two consecutive numbers to have two distinct prime factors are:

        14 = 2  7
        15 = 3  5

        The first three consecutive numbers to have three distinct prime factors are:

        644 = 2^2 7  23
        645 = 3  5  43
        646 = 2  17  19.

        Find the first four consecutive integers to have four distinct prime factors.
        What is the first of these numbers? 
    """
    for n in arange(10**7):
        if n%1e4==0: print n

        p1 = primeFactors(n, 4)
        if p1 != None:
            p1 = unique(asarray(flat(listit(p1))))
            p2 = primeFactors(n+1, 4)
            if p2 != None and len(p1) == 4:
                p2 = unique(asarray(flat(listit(p2))))
                p3 = primeFactors(n+2, 4)
                if p3 != None and len(p2) == 4:
                    p3 = unique(asarray(flat(listit(p3))))
                    p4 = primeFactors(n+3, 4)
                    if p4 != None and len(p3) == 4:
                        p4 = unique(asarray(flat(listit(p4))))
                        if len(p4) == 4:
                            print n
                            break
def capsInSeries(n):
    """ finds the number of capacitances in series for n caps """
    if n == 1:
        return 1
        # -x-
    if n == 2:
        return 2+1
        # -x- -x-, p(x,x)
    if n == 3:
        return 4+2+1
        # -x- -x- -x-, 
        # -x- p(x,x)
        # p(x+x, x)
        # p(x,x,x)
    if n >= 4:
        caps = 0
        for c in arange(n-2)+2:
            caps += capsInSeries(c)
        return caps

def primes(n):
    """ generates primes below n, by a sieve of erathonessasf"""
    list = np.arange(0,n)
    prime = np.zeros(n) + 1
    # 1 for prime and 0 for not
    for k in list:
        if prime[1] == 0 or k == 0 or k == 1:
            continue
        mul = 2
        while mul*k < n:
            prime[mul*k] = 0
            mul += 1

    primes = []
    for k in list:
        if prime[k] == 1:
            primes += [k]
    return primes

def p47():
    # not sure if really 47
    sum = 0
    primeCount = 0
    for n in primes(int(1e7))[7:]:
        if n>1e3:
            #print "----------"
            count = 0
            if is_prime(n): count += 1
            numberStay = n
            for div in arange(len(str(n)))+1:
                n = n//10
                #print n, numberStay
                if is_prime(n) and n != 0:
                    count += 1
                elif n==0: n = 0
                else: break

            n = str(numberStay)
            # before and after: n is a str
            for mod in arange(len(str(n))):
                if n[mod:] != '':
                    n = str(int(n[mod:]))
                    #print n, numberStay

                    if is_prime(int(n)): count += 1
                    elif n == 0: n=0
                    else: continue

                    if count == (len(str(numberStay))-1)*2 + 1:
                        sum += numberStay
                        primeCount += 1
                        print  "numberStay =", numberStay,"... sum =", sum,"... primeCount =", primeCount

def p81():
    # problem 81
    # we want to get from the upper left to lower right of a matrix
    import random
    np.random.seed(42)
    matrix = np.random.rand(10*10) * 100
    matrix = np.round(matrix)
    matrix.shape = (10, 10)
    print matrix

    matrix[9,9-1] = matrix[9,9] + matrix[9, 9-1]
    matrix[9-1,9] = matrix[9,9] + matrix[9-1, 9]
    # test the two sides...
    x = 8; y = 9
    if matrix[y,x-1] < matrix[y-1,x]:
        matrix[y,x-1] += matrix[y,x]
    else:
        matrix[y-1,x] += matrix[y, x]

    print matrix



def findFinishValue(n):
    # finding the new number...
    if n==0: return 1
    if n==1: return 1
    if n==89: return 89
    # requires dynamic scope
    newNumber = 0
    for i in arange(len(str(n))):
        newNumber += int(str(n)[i])**2
    return findFinishValue(newNumber)

def findNewNumber(n, newNumber):
    newNumber = 0
    for i in arange(len(str(n))):
        newNumber += int(str(n)[i])**2
    return newNumber


def p92():
    """ 
        A number chain is created by continuously adding the square of the digits in a number to form a new number until it has been seen before.

        For example,

        44  32  13  10  1  1
        85  89  145  42  20  4  16  37  58  89

        Therefore any chain that arrives at 1 or 89 will become stuck in an endless loop. What is most amazing is that EVERY starting number will eventually arrive at 1 or 89.

        How many starting numbers below ten million will arrive at 89?
    """
    # n[number] = numbersFinishValue
    finish = zeros(700, dtype=int)-1
    finish[0] = 1
    finish[1] = 1
    finish[89] = 89
    for n in arange(700):
        finish[n] = findFinishValue(n)

    count = 0; n = 0
    while n<10e6:
        n += 1
        i = findNewNumber(n)
        if finish[i] == 89: count += 1
        if n%1e5 == 0: print n/1e6
    print count


def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:      
        a, b = b, a % b
    return a

def p71():
    """
        Problem 71
        Consider the fraction, n/d, where n and d are positive integers. If nd and HCF(n,d)=1, it is called a reduced proper fraction.

        If we list the set of reduced proper fractions for d  8 in ascending order of size, we get:

        1/8, 1/7, 1/6, 1/5, 1/4, 2/7, 1/3, 3/8, 2/5, 3/7, 1/2, 4/7, 3/5, 5/8, 2/3, 5/7, 3/4, 4/5, 5/6, 6/7, 7/8

        It can be seen that there are 3 fractions between 1/3 and 1/2.

        How many fractions lie between 1/3 and 1/2 in the sorted set of reduced proper fractions for d  12,000?
    """

    count = 0
    for d in arange(1,12e1+1):
        #if d % 1e2 == 0: print d/1e3
        for n in arange(d//3.35, d//1.93):
            if gcd(n, d) == 1 and n/d < 1/2 and n/d > 1/3:
                count += 1



"""
    Problem 65:
    The square root of 2 can be written as an infinite continued fraction.

    sqrt(2) = 1 + 1/(1 + 1/(2 + 1/(2 + 1/(2 +...))))

                                    2 + ...
    The infinite continued fraction can be written, 2 = [1;(2)], (2) indicates that 2 repeats ad infinitum. In a similar way, 23 = [4;(1,3,1,8)].

    It turns out that the sequence of partial values of continued fractions for square roots provide the best rational approximations. Let us consider the convergents for 2.

    Hence the sequence of the first ten convergents for 2 are:


    1, 3/2, 7/5, 17/12, 41/29, 99/70, 239/169, 577/408, 1393/985, 3363/2378, ...
    What is most surprising is that the important mathematical constant,
    e = [2; 1,2,1, 1,4,1, 1,6,1 , ... , 1,2k,1, ...].

    The first ten terms in the sequence of convergents for e are:

    2, 3, 8/3, 11/4, 19/7, 87/32, 106/39, 193/71, 1264/465, 1457/536, ...
    The sum of digits in the numerator of the 10th convergent is 1+4+5+7=17.

    Find the sum of digits in the numerator of the 100th convergent of the continued fraction for e.
"""

# first approximate e



# problem 65
# first, approximate sqrt(2)
# sq = 1
# sq = 1 + 1/2
# sq = 1 + 1/(2 + 1/2)

# sq = 1 + 1/(2 + 1/(2 + 1/2))
# sq = 1 + 1

# starting from the bottom    
sq = 0
for k in arange(1e1):
    sq = 1/(sq + 2)
    num = k*2
    den = 1
    print num, den, num/den
print "---"
sq += 1


# now, sqrt(23) = [4; (1 3 1 8)]
sq = 0
for k in arange(10):
    sq = 1/(sq + 1/3 + 1/8)
sq += 4

# now e
# e = [2; 1,2,1, 1,4,1, 1,6,1 , ... , 1,2k,1, ...]

ee = 0
for k in arange(10, 0, -1):
    ee = 1/(ee + 1/(2*k + 1))
    num = 2*k + 1
    # now we have to simplify this into a fraction (easy with sage)

# ee = 2
# ee = 2 + 1/(2+1)
# ee = 2 + 1/(2+1 + 1/(4+1))
# ee = 2 + 1/(2+1 + 1/(4+1 + 1/(6+1)))
# ee = 2 + 1/(2+1 + 1/(4+1 + 1/(6+1 + 1/(8+1))))

print "ee =", ee
ee += 0

def mul8642(n, start=0):
    answer = 1
    for i in arange(start, n)+1:
        if i%2 == 1: continue
        else: answer *= i
    return answer













