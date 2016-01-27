h=1/2; 
x=[1-h/2 1+h/2];
y=1./(1+25*x.^2);
c=polyfit(x,y,1);
error=polyval(c,1)-1/26;

for m=1:1:5
    x=[1-h/2 1+h/2];
    y=1./(1+25*x.^2);
    c=polyfit(x,y,1);
    h
    error=polyval(c,1)-1/26
    h=h/2;
end