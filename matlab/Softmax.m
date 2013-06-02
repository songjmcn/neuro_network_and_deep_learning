function out=Softmax(data)
    [n,m]=size(data);
    h=exp(data);
    s=sum(h,1);
    out=h./repmat(s,n,1);
    %out=bsxfun(@rdivide,h,sum(h));
end


