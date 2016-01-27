function [ p ] = mypolyval( P, X )
n=length(P)-1;
p=P(1);
for i=1:n
    p=p.*X+P(i+1);
end

