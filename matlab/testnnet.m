function error = testnnet( nnet,samples,labels )
%test nnet
n_l=length(nnet.w);
input=samples;
m=size(samples,2);
input=samples;
out=0;
for i=1:n_l
    a=nnet.active{i}.fun(nnet.w{i}*input+repmat(nnet.b{i},1,m));
    input=a;
    out=a;
end
[pre_pro,pre_labels]=max(out);
error=mean(pre_labels~=labels);
end

