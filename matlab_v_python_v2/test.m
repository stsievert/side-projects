clc; clear all; close all;

tic;

% simple for loop
ans = 0;
for i=0:1e6,
    ans = ans + i;
end
forLoop = toc;
display(forLoop)

% arange, sum
tic;
x = 0:1e8;
y = sum(x);
vecFor = toc;
display(vecFor)

% 
tic;
n = 524;
x = rand(n);
[U,S,V] = svd(x);
svdTime = toc;
display(svdTime)

tic;
x = 0:1e7;
y = cumsum(x);
cumSumTime = toc;
display(cumSumTime)


tic;
a = 1:1e3;
b = 1:1e3;
[A, B] = meshgrid(a, b);
c2 = A.^2 + B.^2;
C = sqrt(c2);
i = find(A+B+C == 1000);
ans = A(i(1))*B(i(1))*C(i(1));
euler_time = toc;
display(euler_time)
