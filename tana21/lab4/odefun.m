function [ f ] = odefun(x,y,z)
%ODEFUN Summary of this function goes here
%   Detailed explanation goes here
kp=0.1;
ki=0.125;

A=[-5*10^-5 1 0 ; -kp -1 -ki ; 1 0 0 ];
b=[-1*10^-3; 1*10^-3; 0];

f=A*y+b;


end

