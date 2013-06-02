function [ net ] = trainsoftmax( net,samples,labels,eta )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[n,m]=size(samples);
w=net.w;
b=net.b;
lambda=1e-4;
%Ç°Ïò´«²¥
tmp=w*samples+repmat(b,1,m);
M=bsxfun(@minus,tmp,max(tmp,[],1));
out=Softmax(M);
targetOut=zeros(size(out));
for i=1:m
    targetOut(labels(i),i)=1;
end
dout=-(targetOut-out);
dw=1/m.*dout*samples'+lambda.*w;
db=1/m.*sum(dout,2);
w=w-eta.*dw;
b=b-eta.*db;
net.w=w;
net.b=b;
end
