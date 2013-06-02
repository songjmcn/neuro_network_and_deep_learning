function sigm = sigmoid( x )
%激活函数
 %S型激活函数
 %sigm=(1./(1+exp(-x)));
 %双S型激活函数
 %sigm=(2 ./(1+exp(-x)))-1;
 %tanh型激活函数
 sigm=tanh(x);
end

