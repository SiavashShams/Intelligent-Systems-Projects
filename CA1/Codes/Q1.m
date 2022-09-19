clear 
clc

x=[1;1];
alpha_max = 1;
iter = 1;
tol = 10^(-6);
max_iter = 10000;
f_obj = zeros(max_iter, 1);
alpha = alpha_max * ones(max_iter, 1);
alpha(1) = 0;
xx=ones(max_iter,2);
while norm(f_grad(x)) > tol && (iter < max_iter)  %checking norm of gradient 
  f_obj(iter) = f_func(x);
  iter = iter + 1;
  p=f_grad(x);
  alpha(iter)=step_size(x,p);
  x = x + alpha(iter) * p;
  xx(iter,:)=x;
end
f_obj(iter) = f_func(x);
f_obj = f_obj(1:iter);
alpha = alpha(1:iter);
xx=xx(1:iter,:);

%Find optimal step alpha in each iteration
function alph = step_size(x_k, p_k)
syms alph
f_x_k = f_func(x_k+alph*p_k);
df=diff(f_x_k);
sol=solve(df,alph);
alph=double(sol);
end

% Function to minimize and its gradient 

function f = f_func(x)
f = 3*x(1)^2+12*x(1)+8*x(2)^2+8*x(2)+6*x(1)*x(2);
end

function grad = f_grad(x)
grad = -[6*x(1)+6*x(2)+12; 16*x(2)+6*x(1)+8];
end

