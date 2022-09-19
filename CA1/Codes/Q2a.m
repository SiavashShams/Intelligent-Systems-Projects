clear 
clc
%Gradient Descent with fixed step size
x=[5;5];
%x=[-5,5];  %second point -> uncomment for use
alpha_max = 1;
iter = 1;
tol = 1;
max_iter = 1000;
f_obj = zeros(max_iter, 1);
alpha = alpha_max * ones(max_iter, 1);
xx=ones(max_iter,2);
i=0;
xx(1,:)=x;
alpha=0.01;
while norm(f_grad(x)) > tol && (iter < max_iter)  %checking norm of gradient 
  f_obj(iter) = f_func(x);
  iter = iter + 1;
  p=f_grad(x);
  x = x + alpha * p;
  xx(iter,:)=x;
end
f_obj(iter)=f_func(x);
f_obj = f_obj(1:iter);

xx=xx(1:iter,:);
f_obj(iter-1)


% Function to minimize and its gradient 

function f = f_func(x)
f = x(1)^2-10*x(2)*cos(0.2*pi*x(1))+x(2)^2-15*x(1)*cos(0.4*pi*x(2));
end

function grad = f_grad(x)
grad = -[2*x(1)+2*pi*x(2)*sin(0.2*pi*x(1))-15*cos(0.4*pi*x(2));
     2*x(2)+6*pi*x(1)*sin(0.4*pi*x(2))-10*cos(0.2*pi*x(1))];
end

