i=0;
clf
x=(0:1:49);
hold on
X = [];
E = [];
P = [];
data=[];

x=0;
j=0;
while x<=10
    x=x+0.1;
    if floor(x) <= abs(x*1.0001)
        x=x+0.1;
    end
    j=j+1;
    i=0;
    X = [];
E = [];
P = [];
    while i<100
        i=i+1;
        %x = rand(1)*100;
        p = (1/100)*i;
        e = (x^p - S2pow(x,p)) / x^p;
        X = [X x];
        P = [P p];
        E = [E e];
        %plot3(x,p,e);
    end
    data = [data; P; E];

    
end

for k=1:90
    rndcolor = [rand, rand, rand];
    str = sprintf('x=%d', 0.1*k);
    text(0.3, data(2*k-1,30), str);
    
    
	plot(data(2*k-1,:), data(2*k,:), 'color', rndcolor); 
    
    %xstext(0.3, data(2*k,30), str);

end











%sort(data);
%scatter(X,P,20,E,'*');
%plot3(X,P,E)






% while i<100
%     i=i+1
%     p=i/1.2;
%     run = S2pow(3, p);
%     run = run / (3^p);
%     
%     if length(run) == 1;
%         runs = run;
%         for j=1:50
%             runs = [runs run];
%         end
%         run = runs;
%     end
%     
%     run = run(1,1:50);
%     
%     
%     plot(x, run)
%     %string = sprintf('i=%d', i);
%     %text(0,run(1),string,...
%     % 'HorizontalAlignment','left')
% end


