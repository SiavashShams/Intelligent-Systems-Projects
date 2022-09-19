clear 
clc
x=[0;0];
iter = 1;
tol = 1;
max_iter = 1000;
f_obj = zeros(max_iter, 1);
f_obj_m1=zeros(max_iter, 1);
alpha = 1 * ones(max_iter, 1);
alpha(1) = 0;
alpha_m1 = 1 * ones(max_iter, 1);
alpha_m1(1) = 0;
xx=ones(max_iter,2);
xx_m1=ones(max_iter,2);
i=0;
xx(1,:)=x;
xx_m1(1,:)=x;
errf_m1=[];
errf=[];
while norm(f_grad(x)) > tol && iter < max_iter  %checking norm of gradient
  errf_m1(iter)=norm(f_grad(x));
  iter = iter + 1;
  p=f_grad(x);
  alpha_m1(iter)=step_size_m1(x,p);
  x = x + alpha_m1(iter) * p;
  xx_m1(iter,:)=x;
  f_obj_m1(iter) = f_func(x);
end
f_obj_m1 = f_obj_m1(1:iter);
alpha_m1 = alpha_m1(1:iter);

xx_m1=xx_m1(1:iter,:);
f_obj_m1(iter)

x=[0;0];
iter=1;
while norm(f_grad(x)) > tol && iter < max_iter  %checking norm of gradient 
  errf(iter)=norm(f_grad(x));
  p=f_grad(x);
  if (iter <=2)
      alpha(iter)=step_size(x,p,0);
  else
  alpha(iter)=step_size(x,p,f_obj(iter-2));
  end
  x = x + alpha(iter) * p;
  f_obj(iter) = f_func(x);
  iter = iter + 1;
  xx(iter,:)=x;
end

f_obj = f_obj(1:iter-1);
alpha = alpha(1:iter-1);

xx=xx(1:iter-1,:);
f_obj(iter-1)







%Find optimal step alpha in each iteration using analytical method
function alph = step_size_m1(x_k, p_k)
syms alph
f_x_k = f_func(x_k+alph*p_k);
df=diff(f_x_k);
sol=vpasolve(df,alph);
alph=double(sol);
end

%Find optimal step alpha in each iteration using Armijo method

function alpha = step_size(x_k, p_k,f_k_p)
if(f_k_p ==0)
    alpha=1;
else
    alpha = 2*(f_func(x_k)-f_k_p)/(-f_grad(x_k)'*p_k);
end
c=0.01;
beta=0.2;
f_x_k = f_func(x_k);
while (f_func(x_k + alpha * p_k) > ...
      f_x_k + c * alpha * -f_grad(x_k)' * (p_k/norm(p_k)))
   alpha = beta * alpha;
end
end

% Function to minimize and its gradient 

function f = f_func(x)
f = x(1)^2-10*x(2)*cos(0.2*pi*x(1))+x(2)^2-15*x(1)*cos(0.4*pi*x(2));
end

function grad = f_grad(x)
grad = -[2*x(1)+2*pi*x(2)*sin(0.2*pi*x(1))-15*cos(0.4*pi*x(2));
    2*x(2)+6*pi*x(1)*sin(0.4*pi*x(2))-10*cos(0.2*pi*x(1))];
end

