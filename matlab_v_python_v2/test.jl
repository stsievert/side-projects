
function forLoopTime()
    x = 0;
    for i=0:1e6
        x = x + i;
    end
end

function vecForTime()
    x = 1:1e8;
    y = sum(x);
end

function svdTime()
    n = 524;
    x = randn(n,n);
    y = svd(x);
end

function cumSumTime()
    x = 1:1e7;
    y = cumsum(x);
end
function ndgrid{T}(v1::AbstractVector{T}, v2::AbstractVector{T})
    m, n = length(v1), length(v2)
    v1 = reshape(v1, m, 1)
    v2 = reshape(v2, 1, n)
    (repmat(v1, 1, n), repmat(v2, m, 1))
end
function euler()
    a = linspace(1,1000,1000);
    b = a;
    A,B = ndgrid(a,b)
    #A = a;
    #B = b;
    c2 = A.^2 + B.^2;
    C = sqrt(c2);
    #i = findnz(A+B+C - 1000)
end

forLoopTime();
vecForTime();
svdTime();
cumSumTime();

tic(); forLoopTime(); toc();
tic(); vecForTime(); toc();
tic(); svdTime(); toc();
tic(); cumSumTime(); toc();

tic(); euler(); toc();
