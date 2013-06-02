function net = trainBp( net,samples,labels,eta)
%训练BP网络

lambda=1e-4;
%前向传播
[n,m]=size(samples);
w1=net.hidden.w;
b1=net.hidden.b;
w2=net.out.w;
b2=net.out.b;
z2=w1*samples+repmat(b1,1,m);
a2=sigmoid(z2);
z3=w2*a2+repmat(b2,1,m);
a3=Softmax(z3);
[pre,pre_labels]=max(a3);
pre_labels=pre_labels';
error=mean(pre_labels~=labels);
%fprintf('训练错误率为%g%%\n',error*100);
%反向传播
targetOut=zeros(size(a3));
for ii=1:m
    targetOut(labels(ii),ii)=1;
end
d3=-(targetOut-a3);
d2=(w2'*d3).*dsigmoid(z2);
dw1=(1/m).*d2*samples'+lambda*w1;
dw2=(1/m).*d3*a2'+lambda*w2;
db1=(1/m).*sum(d2,2);
db2=(1/m).*sum(d3,2);
net.hidden.w=w1-eta.*dw1;
net.hidden.b=b1-eta.*db1;
net.out.w=w2-eta.*dw2;
net.out.b=b2-eta.*db2;
end

