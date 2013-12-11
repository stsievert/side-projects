
# plain for
start = proc.time()
ans = 0
for (i in 1:1e6) ans = ans + i
print(proc.time() - start)

# vecFor
start = proc.time()
x = seq(1e8)
y = sum(as.numeric(x))
print(proc.time() - start)

# svd
start = proc.time()
n = 524
x = matrix( rnorm(n*n,mean=0,sd=1), n, n) 
d = svd(x)
print(proc.time() - start)

# cumSum
start = proc.time()
x = seq(1e7)
y = cumsum(as.numeric(x))
print(proc.time() - start)

meshgrid <- function(a,b) {
  list(
       x=outer(b*0,a,FUN="+"),
       y=outer(b,a*0,FUN="+")
       )
} 
a = seq(1e3)
b = a
list[A,B] = meshgrid(a,b)

    #a = linspace(1,1000,1000);
    #b = a;
    #A,B = ndgrid(a,b)
    ##A = a;
    ##B = b;
    #c2 = A.^2 + B.^2;
    #C = sqrt(c2);
    #i = findnz(A+B+C - 1000)
