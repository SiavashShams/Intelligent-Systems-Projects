clc
clear
f= @(x) (x(1)^2-10*x(2)*cos(0.2*pi*x(1))+x(2)^2-15*x(1)*cos(0.4*pi*x(2)));
x=[0,0];
fx=feval(f,x);
f0=fx;
iter=0;
maxtemps=100;
while iter<maxtemps
    T=iter/maxtemps*100;     
    for k=0:1000        
        a=normrnd(x,T); %Guassian random number to update x
        xx=x+a;
        fxx=feval(f,xx);
        del=fxx-fx;       
        if (exp(-del/T)> rand(1)  )         
            x=xx;
            fx=fxx;
        end        
        if fxx < f0     %Update minimum point
        x0=xx;
        f0=fxx;
        end   
    end
    iter=iter+1;
end
fprintf('Mimimum found for the function equals %f\n', f0)
fprintf('Vector that results in minimum is %f %f \n', x0)

