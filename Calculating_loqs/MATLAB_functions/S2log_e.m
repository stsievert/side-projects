function [ answer ] = S2log_e( n )
% This function needs to work for any x, not just abs(x) < 1
% Solution: integrate (1/x) from 1 to x.
% Integration:
%   dx (box width) porportional to 1/slope = x*x
%   
%    
%     
%     
    % find m less than 1.
    m=n;
    j=0;
    while floor(m) > 0
       m = m/2;
       j=j+1;
    end
    
    % log(n) = log(m) + i*log(2)
    % We'll find log(m) by using Taylor series for ln(x+1)
    % We'll find log(2) by integrating.
    
    % Find log(m)
    s = m-1;
    log_m = 0;
    k=0;
    term = 1;
    while abs(term) > 1e-20
        k=k+1;
        term = S2intpow2(s, k)/k;
        if mod(k, 2) == 0
            term = -1*term;
        end
        log_m = term + log_m;
    end
    
    % Now, find log(2)
    log_2 = S2log_integration(2);
    
    log_n = log_m + j*log_2;
    answer = log_n;
    

end




function [ answer ] = S2log_integration( n )
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
        dx = (x*x)/10000;
        dx_over_x = dx/x;
        ln_n = ln_n + dx/x;
        x = x + dx;
    end
    answer = ln_n;

end



function [ answer ] = S2intpow2( x, p)
    answer = 1;
    for i=1:p
        answer = answer*x;
    end  
end