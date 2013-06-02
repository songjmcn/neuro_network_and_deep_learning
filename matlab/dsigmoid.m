function out = dsigmoid( x )
%激活函数的导数
%S型激活函数
%tmp=sigmoid(x);
%out=tmp.*(1-tmp);
%双S型激活函数
%tmp=exp(-x);
%out=(2.*tmp)./((1+tmp).*(1+tmp));
%tanh型
tmp=sigmoid(x);
out=1-tmp.*tmp;
end

