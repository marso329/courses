con=2.00;
format long
%root in 1 and 1000
f=@(x) x.^2-1001*x+1000;
[x,it,x_arr]=mysol(f,-10,10,0.0000000001);
convergence1=zeros(1,length(x_arr)-1);
for i=1:length(x_arr)-1
convergence1(i)=abs(x_arr(i+1)-1)/abs(x_arr(i)-1)^con;
end
for i=1:length(convergence1)-1
if convergence1(i+1)<convergence1(i)
    disp('fail')
end
end

f=@(x) 3*x+sin(x)-exp(x);
[x,it,x_arr]=mysol(f,0,1,0.0000000000);
convergence2=zeros(1,length(x_arr)-1);
for i=1:length(x_arr)-1
convergence2(i)=abs(x_arr(i+1)-0.360421702960324)/abs(x_arr(i)-0.360421702960324)^con;
end

for i=1:length(convergence2)-1
if convergence2(i+1)<convergence2(i)
    disp('fail')
end
end

f=@(x) cos(x)-x*exp(x);
[x,it,x_arr]=mysol(f,1,2,0.0000000000001);
convergence3=zeros(1,length(x_arr)-1);
for i=1:length(x_arr)-1
convergence3(i)=abs(x_arr(i+1)-0.517757363682458)/abs(x_arr(i)-0.517757363682458)^con;
end

for i=1:length(convergence3)-1
if convergence3(i+1)<convergence3(i)
    disp('fail')
end
end

f=@(x) x.^4-x-10;
[x,it,x_arr]=mysol(f,-2,-1.5,0.0000001);
convergence4=zeros(1,length(x_arr)-1);
for i=1:length(x_arr)-1
convergence4(i)=abs(x_arr(i+1)+1.697471880844153)/abs(x_arr(i)+1.697471880844153)^con;
end
for i=1:length(convergence4)-1
if convergence4(i+1)<convergence4(i) && convergence4(i+1)~=0 
    disp('fail')
end
end

convergence4