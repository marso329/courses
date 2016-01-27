runtime=0.05;
elapsedtime=0.0;
n=1000;
while elapsedtime<=runtime
A=triu(rand(n,n));b=rand(n,1);tic,A\b;elapsedtime=toc;
n=n+100;
end
n=n-100;
elapsedtime=0.0;
while elapsedtime<=runtime
A=triu(rand(n,n));b=rand(n,1);tic,A\b;elapsedtime=toc;
n=n+10;
end
n=n-10;
elapsedtime=0.0;
while elapsedtime<=runtime
A=triu(rand(n,n));b=rand(n,1);tic,A\b;elapsedtime=toc;
n=n+1;
end
disp(['to calculate a linear system with ',num2str(n),' variables takes ',num2str(elapsedtime),' seconds'])
n=n*2;
A=triu(rand(n,n));b=rand(n,1);tic,A\b;elapsedtime=toc;
disp(['to calculate a linear system with ',num2str(n),' variables takes ',num2str(elapsedtime),' seconds'])
n=n*2;
A=triu(rand(n,n));b=rand(n,1);tic,A\b;elapsedtime=toc;
disp(['to calculate a linear system with ',num2str(n),' variables takes ',num2str(elapsedtime),' seconds'])