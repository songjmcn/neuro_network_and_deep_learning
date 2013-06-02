function error = testBp( net,samples,labels)
%²âÊÔBPĞÔÄÜ
[n,m]=size(samples);
w1=net.hidden.w;
b1=net.hidden.b;
a=sigmoid(w1*samples+repmat(b1,1,m));
w2=net.out.w;
b2=net.out.b;
out=Softmax(w2*a+repmat(b2,1,m));
[pre,pre_labels]=max(out);
pre_labels=pre_labels';
error=mean(pre_labels~=labels);
end

