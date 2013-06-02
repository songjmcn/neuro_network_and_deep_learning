function error = testSoftmax( net,samples,labels )
%测试softmax
%net 网络的定义（参数）
%samples 测试样本
%labels 测试样本的标记
[n,m]=size(samples);
w=net.w;
b=net.b;
out=w*samples+repmat(b,1,m);
M=bsxfun(@minus,out,max(out,[],1));
h=Softmax(M);
[pre,pre_labels]=max(h);
pre_labels=pre_labels';
error=mean(pre_labels~=labels);

end

