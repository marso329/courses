n=[1 2 4 8 16 32 64 128 256 512 1024 2048 4096];
data=zeros(7,length(n));
i=1;
for x=n
    %e^x [0 1]
    data(1,i)=abs((exp(1)-exp(0))-trapezoid(@exp,0,1,n(i)));
    
    %3*x^2+2*x+1 [0 1]
    f=@(x) 3*x^2+2*x+1;
    f_int=@(x) x^3+x^2+x;
    data(2,i)=abs((f_int(1)-f_int(0))-trapezoid(f,0,1,n(i)));
    
    %2*x+2 [0 1]
    f=@(x) 2*x+2;
    f_int=@(x) x^2+2*x;
    data(3,i)=abs((f_int(1)-f_int(0))-trapezoid(f,0,1,n(i)));
    
    %5*x^4+3*x^3+4*x^2+x+15 [0 1]
    f=@(x) 5*x^4+3*x^3+4*x^2+x+15;
    f_int=@(x) x^5+(3/4)*x^4+(4/3)*x^3+(1/2)*x^2+15*x;
    data(4,i)=abs((f_int(1)-f_int(0))-trapezoid(f,0,1,n(i)));
    
    %4/(1+x^2) [0 1]
    f=@(x) 4/(1+x^2);
    data(5,i)=(4*(atan(1)-atan(0)))-trapezoid(f,0,1,n(i));

    %x^(1/2) [0 1]
    f=@(x) x^(1/2);
    f_int=@(x) (2/3)*x^(3/2);
    data(6,i)=abs((f_int(1)-f_int(0))-trapezoid(f,0,1,n(i)));

    %sin(x) [0 2*pi]    
    f=@(x) sin(x);
    f_int=@(x) -cos(x);
    data(7,i)=abs((f_int(2*pi)-f_int(0))-trapezoid(f,0,2*pi,n(i)));
    
    i=i+1;
end
clf
hold on;
for j=[1:1:6]
plot(log(n),log(data(j,:)))
log_data=log(data(j,:));
log_axis=log(n);
(log_data(length(data(j,:)))-log_data(1))/(log_axis(length(log_axis))-log_axis(1))
end
xlabel('log(h)')
ylabel('log(error)')
legend('e^x','3*x^2+2*x+1', '2*x+2','5*x^4+3*x^3+4*x^2+x+15','4/(1+x^2)','x^{1/2}')

data;