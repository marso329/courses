function x = newton(f, fp, x, n)
for k = 1:n
    x = x - f(x)/fp(x);
end