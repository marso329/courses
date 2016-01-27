n=4
x=-1:2/n:1;
y=1./(1+25*x.^2);
hold;
xx=-1:0.001:1;
for m=1:n+1
    c=polyfit(x,y,m);
    yy=polyval(c,xx);
    plot(xx,yy,'*');
    plot(x,y,'*');
    m
    pause(1.0);
    
end