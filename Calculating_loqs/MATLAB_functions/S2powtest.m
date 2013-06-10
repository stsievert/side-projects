clf;
clear;
b= 2  ;
p= 1/5 ;
run = S2pow(b, p);
x = (0:1:length(run)-1);
real = ones(1,length(run));
real = real*(b^p);
plot(x, run, 'r-')
hold on;
plot(x, real, 'b-')