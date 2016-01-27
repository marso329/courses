[tv1, f1]=ode23('odefun',[0 40],[-1; 0.001;0]);
y=f1(:,1);
plot(tv1,y);

