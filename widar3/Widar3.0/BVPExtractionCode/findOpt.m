% %方式一：分成多个文件
% function f=fminxy(t)
% x=t(1);y=t(2);
% f=x*x*x-y*y*y+2*x*x+x*y;
% end
% %%以上代码生成fminxy文件
% function [c d]=fcontr(t)
% x=t(1);y=t(2);
% c=x*x+y*y-6;
% d=x*y-2;
% end
% %%以上文件生成fcontr文件
% >> [x,fval,exitflag]=fmincon('fminxy',[1 2],[],[],[],[],[],[],'fcontr')
% %fmincon(	fun,	x0, A, b,Aeq,beq,lb,ub, nonlcon) A是初始值，fmincon的初始值x0可以任意取，只要保证为实数就行。
% x =
%     0.8740     2.2882      %函数最小值时x,y的值
% fval =
%    -7.7858                        %函数最小值
% exitflag =
%      1                               %一阶最优性条件满足容许范围，既是结果正确

%方式二：整合成一个文件
function  x= findOpt(a,b) 
x=fmincon(@(t) fminxy(t),[a b],[],[],[],[],[],[],@(t) fcontr(t));
end
 
function f=fminxy(t)
x=t(1);y=t(2);
f=x*x*x-y*y*y+2*x*x+x*y;
end
 
function [c d]=fcontr(t)
x=t(1);y=t(2);
c=x*x+y*y-6;
d=x*y-2;
end
%以上代码生成findOpt文件
% >> x=findOpt(2,2)
