function[x,it,x_arr] = mysol(f,x1 ,x2 ,error)
x = (x1*f(x2) - x2*f(x1))/(f(x2) - f(x1));
it=1;
x_arr=[];
while abs(f(x2)) > error
    it=it+1;
    x1 = x2;
    x2 = x;
    x = (x1*f(x2) - x2*f(x1))/(f(x2) - f(x1));
    x_arr=[x_arr x];
end
end