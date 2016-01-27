%e^x [0 1]
exact=exp(1)-exp(0);
trapets=trapezoid(@exp,0,1,1000);
error=exact-trapets;
disp(['e^x integrated from 0 to 1 using trapezoid with 1000 intervalls is: '...
    ,num2str(trapets),' and the error is : ',num2str(error)])

%3*x^2+2*x+1 [0 1]
f=@(x) 3*x^2+2*x+1;
f_int=@(x) x^3+x^2+x;
exact=f_int(1)-f_int(0);
trapets=trapezoid(f,0,1,1000);
error=exact-trapets;
disp(['3*x^2+2*x+1 integrated from 0 to 1 using trapezoid with 1000 intervalls is: '...
    ,num2str(trapets),' and the error is : ',num2str(error)])

%2*x+2 [0 1]
f=@(x) 2*x+2;
f_int=@(x) x^2+2*x;
exact=f_int(1)-f_int(0);
trapets=trapezoid(f,0,1,1000);
error=exact-trapets;
disp(['2*x+2 integrated from 0 to 1 using trapezoid with 1000 intervalls is: '...
    ,num2str(trapets),' and the error is : ',num2str(error)])

%5*x^4+3*x^3+4*x^2+x+15 [0 1]
f=@(x) 5*x^4+3*x^3+4*x^2+x+15;
f_int=@(x) x^5+(3/4)*x^4+(4/3)*x^3+(1/2)*x^2+15*x;
exact=f_int(1)-f_int(0);
trapets=trapezoid(f,0,1,1000);
error=exact-trapets;
disp(['5*x^4+3*x^3+4*x^2+x+15 integrated from 0 to 1 using trapezoid with 1000 intervalls is: '...
    ,num2str(trapets),' and the error is : ',num2str(error)])

%4/(1+x^2) [0 1]
f=@(x) 4/(1+x^2);
exact=4*(atan(1)-atan(0));
trapets=trapezoid(f,0,1,1000);
error=exact-trapets;
disp(['4/(1+x^2) integrated from 0 to 1 using trapezoid with 1000 intervalls is: '...
    ,num2str(trapets),' and the error is : ',num2str(error)])

%x^(1/2) [0 1]
f=@(x) x^(1/2);
f_int=@(x) (2/3)*x^(3/2);
exact=(f_int(1)-f_int(0));
trapets=trapezoid(f,0,1,1000);
error=exact-trapets;
disp(['x^(1/2) integrated from 0 to 1 using trapezoid with 1000 intervalls is: '...
    ,num2str(trapets),' and the error is : ',num2str(error)])

%sin(x) [0 2*pi]
f=@(x) sin(x);
f_int=@(x) -cos(x);
exact=(f_int(2*pi)-f_int(0));
trapets=trapezoid(f,0,2*pi,1000);
error=exact-trapets;
disp(['sin(x)) integrated from 0 to 2*pi using trapezoid with 1000 intervalls is: '...
    ,num2str(trapets),' and the error is : ',num2str(error)])

%test accuracy
%n=[1 2 4 8 16 32 64 128 256 512 1024 2048 4096];
n=1:1:100;
errors=[];
for x=n
exact=exp(1)-exp(0);
trapets=trapezoid(@exp,0,1,x);
error=abs(exact-trapets);
errors=[errors,error];
end
plot(n,errors);
%axis([0,100,0,0.05]);
xlabel('Intervals')
ylabel('Error')
legend('e^x')

%test time complexity
n=[1 2 4 8 16 32 64 128 256 512 1024 2048 4096];
times=[];
for x=n
    tic;
trapets=trapezoid(@exp,0,1,x);
times=[times,toc];
end
plot(n,times);
xlabel('Intervals')
ylabel('Time')