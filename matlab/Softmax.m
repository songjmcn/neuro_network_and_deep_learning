function out=Softmax()
    out.fun=@fun;
    out.dfun=@dfun;
    %out=bsxfun(@rdivide,h,sum(h));
end
function out=fun(x)
    [n,m]=size(x);
    h=exp(x);
    s=sum(h,1);
    out=h./repmat(s,n,1);
end
function out=dfun(x)
    out=1;
end

