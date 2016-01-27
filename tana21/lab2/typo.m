x = [0, 1, 1, 0.5, 0, 0, 1];
y = [0, 0, 1, 1, 1, 2, 2];

t = 1:7;

px = csape(t, x);
py = csape(t, y);

tt = 1:0.1:7;

plot(fnval(px, tt), fnval(py, tt)), axis equal