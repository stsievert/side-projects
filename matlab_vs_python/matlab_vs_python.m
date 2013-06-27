
function [y] = main()
clc; clear all; close all;

W = 1024;
H = 1024;
X = rand(W, H);

tic;
Y = dwt2_full(X);
toc;
end


function [y] = dwt(x)

    w = size(x);
    w = max(w);
    x = reshape(x, 1, w);
    y = zeros([1 w]);
 
    for i=1:w/2
        % i = 0. 0, 1
        %   = 1. 2, 3
        %   = 2. 4, 5
        %   = 3. 6, 7
        %   = 4. 8, 9
        y(i)       = (x(2*i-1) + x(2*i))/sqrt(2);
        y(i + w/2) = (x(2*i-1) - x(2*i))/sqrt(2);
        
    end
end

function [y]=idwt(x)
    % [a+b c+d a-b c-d]
    % x = a+b
    % y = a-b
    % a = (x+y)/2
    w = size(x);
    x = reshape(x, 1, max(w));
    w = size(x);
    y = zeros(w);
    w = w(2);
    % 2*1  = 2
    % 2*2  = 4
    % 2*3  = 6
    % [1+2 3+4 5+6 7+8 1-2 3-4 5-6 7-8]
    % [ a   b   c   d   e   f   g  h  ]
    
    
    % 2*1 -1 = 1
    % 2*2 -1 = 3
    % 2*3 -1 = 5
    
    for i=1:w/2
        % 1, 2, 3, 4
        % 5, 6, 7, 8
        y(2*i-1) = (x(i) + x(i + w/2))/2;
        y(2*i)   = (x(i) - x(i + w/2))/2;
        
    end
    y = y*sqrt(2);


end


function [y] = dwt2(x)
    % x is 2D
    l = size(x);
    y = ones(l);
    for i=1:l(1)
        %row = 1:l(1);
        x(:,i) = dwt(x(:,i));
    end
    for i=1:l(2)
        %column = 1:l(2);
        x(i,:) = dwt(x(i,:));
    end
    y = x;
end

function [y] = idwt2(x)
    l = size(x);
    sz = l;
    for i=1:l(1)
        row = 1:l(1);
        x(:,i) = idwt(x(:,i));
    end

    for i=1:l(2)
        row = l(2);
        x(i,:) = idwt(x(i,:));
    end
    y = x;
end
function [y] = dwt2_full(x)
    sz = size(x);
    order = log2(sz(1));
    for i=0:order-1
        upper = bitshift(sz(1), -1*i);
        y = x(1:upper, 1:upper);
        y = dwt2(y);
        x(1:upper, 1:upper) = y;
    end
    y = x;
end
function [y] = idwt2_full(x)
 
    sz = size(x);
    w = sz(1);
    order = log2(sz(1));
    for i=order:-1:1
        y = x(1:bitshift(w,-i+1),1:bitshift(w,-i+1));
        y = idwt2(y);
        x(1:bitshift(w,-i+1),1:bitshift(w,-i+1)) = y; 
    end
    %y = x;
end

