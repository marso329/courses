%% 2.1
roots([1 0 -2])

%% 2.2
fzero(@(x) x.^2 - 2, [0 5])
fzero(@(x) x.^2 - 2, [-5 0])

%% 3.1
x = -1000:0.1:1000;
f1 = @(x) x.^5;
f2 = @(x) x + 1;
clf;
plot(x, f1(x), x, f2(x));
axis([-1000, 1000, -1000, 1000]);
% 1 reell rot

%% 3.2
roots([1 0 0 0 -1 -1])

%% 3.3
f = @(x) x.^5 - (x + 1);
fzero(f, 0)
% Kan ej best?mma komplexa r?tter

%% 4.1
roots([1 0 -2i])

%% 4.2
fzero(@(x) x.^2 - 2i, 0)
% Det blir fel

%% 5.2
f = @(x) x.^2 -2i;
fp = @(x) 2*x;
newton(f, fp, 3, 5)
newton(f, fp, -0.5, 5)

%% 5.3
f = @(x) x.^2 -2;
fp = @(x) 2*x;
newton(f, fp, 1, 5)

%% 5.4
newton(@sin, @cos, 2, 5)
x = -1:0.1:4;
clf;
plot(x, sin(x));

%% 5.5
newton(@sin, @cos, 0, 5)
newton(@sin, @cos, 1, 5)
newton(@sin, @cos, 2, 5)
newton(@sin, @cos, 3, 5)
newton(@sin, @cos, 4, 5)
newton(@sin, @cos, 5, 5)
newton(@sin, @cos, 6, 5)
newton(@sin, @cos, 7, 5)

%% 6.1
x = -5:0.1:5;
f = @(x) sin(x) - 0.1;
clf;
plot(x, sin(x), x, f(x))

%% 6.2
x = -5:0.01:5;
f1 = @(x) sin(10*x);
f2 = @(x) sin(10*x) - 0.1;
clf;
plot(x, f1(x), x, f2(x));