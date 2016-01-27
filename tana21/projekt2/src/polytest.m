%clf;
times=[];
points= [5 10 20 40 80 160 320 640 1280 2560 5120];
for i = points
    N = i;
    x = 1:N;
    y = randi([-20,20], 1, N);
    tic;
    P = mypolyfit(x, y);
    times=[times [toc]];
    toc;

    px = 1:0.1:N;
    py = mypolyval(P, 1:0.1:N);

    %plot(x, y, '*', px, py);
    %pause;
end
plot(points,times);