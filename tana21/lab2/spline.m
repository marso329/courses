n=100;
x=-1:2/n:1;
y=1./(1+25*x.^2);

xx=-1:0.001:1;

    c=csape(x,y);
    yy=fnval(c,xx);
    clf;
    plot(xx,yy,'*');
    hold;
    plot(x,y,'*');