function I =  trapezoid(f,a,b,n)
h=(b-a)/n;
I=(f(a)+f(b))/2;
for k=1:n-1
    I=I+f(a+k*h);
end
I=h*I;
end