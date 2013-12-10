function [ S2answer ] = S2pow( x , p )

%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Alogrithm: (broad steps):
%
%    x^p = n
%    n = x^floor(p) * x^(fract_p)
%    M = x^(fract_p)
%    ln(M) = fract_p * ln(x)
%    Find M that gives answer close to fract_p*ln(x)
%    
%    ans = M * int_pow(x, floor(p))
%
e = 2.718281828;

pow_int=floor(p);
pow_fract= p - pow_int;

ln_of_ans = pow_fract*S2log_e(x);

% Find us the "ans" that gave us ln_of_ans
    if ln_of_ans <= 1
        % lower bound 0, upper bound e.
        guess_low = 0;
        guess_high = e;
    end
    if ln_of_ans > 1
        guess_low = S2intpow(e, floor(ln_of_ans));
        guess_high = S2intpow(e, ceil(ln_of_ans));
    end
   
    iterations = (guess_low+guess_high);
    i=0;
    while 1
        guess_mid = (guess_high+guess_low)/2;
        if i>0
            iterations = [iterations guess_mid];
        end
        if i>200
            break
        end
        i=i+1;
        
        ln_of_mid = S2log_e(guess_mid);
        
        if abs(guess_mid - guess_high) <= 0.00000000000000000000000000001
            break
        end
       
        if ln_of_mid < ln_of_ans
            guess_low = guess_mid;
        else
            guess_high = guess_mid;
        end
        
    end

S2answer = iterations;
S2answer = iterations*S2intpow(x, pow_int);

end


function [ answer ] = S2log_e( n )
% This function needs to work for any x, not just abs(x) < 1
% Solution: integrate (1/x) from 1 to x.
% Integration:
%   dx (box width) porportional to 1/slope = x*x
%   
%    
%     
%     
     x = 1;
    ln_n = 0;
    while x <= n
        % Calculate dx
        dx = (x*x)/1000000;
        dx_over_x = dx/x;
        ln_n = ln_n + dx/x;
        x = x + dx;
    end
    answer = ln_n;

end




function [ answer ] = S2intpow( x, p)
    answer = 1;
    for i=1:p
        answer = answer*x;
    end  
end