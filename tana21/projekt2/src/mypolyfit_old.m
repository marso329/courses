function [ P ] = mypolyfit_old( X,Y )
%MYPOLYFIT Summary of this function goes here
%   Detailed explanation goes here

%number of data points
n=length(X);

%number of fucks given
0;

%created to make a lower triangular matrix with ones easier
temp_matrix(1:(n),1:(n)) = 1;

%creates a lower triangular matrix with ones
newton_matrix=tril(temp_matrix);

%Do newton
% See https://en.wikipedia.org/wiki/Newton_polynomial section 6: main idea
for j = 2:n
    for i=2:j
        if i>2
            newton_matrix(j,i)=newton_matrix(j,i-1);    
        end
        newton_matrix(j,i)=newton_matrix(j,i)*(X(j)-X(i-1));
    end
end
%Solve linear equations
P=newton_matrix\(Y');
P=rot90(P,-1);
end

