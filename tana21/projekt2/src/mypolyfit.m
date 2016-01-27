function [n] = mypolyfit(x,y)
N = length(x)-1;
temp = zeros(N + 1,N + 1);
temp(1:N + 1,1) = y';
% Create difference table
for k=2:N+1
    for m=1:N+2-k
        temp(m,k) = (temp(m + 1,k - 1) - temp(m,k - 1))/(x(m + k - 1)- x(m));
    end
end
% First row in difference table (def 5.4.1)
a = temp(1,:);
n = a(N+1);
% Nested multiplication for coefficients
for k = N:-1:1 
n = [n a(k)] - [0 n*x(k)];
end