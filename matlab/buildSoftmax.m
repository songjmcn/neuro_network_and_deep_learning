function softmax= buildSoftmax( n_in,n_out )
%初始化softmax classifier参数
%n_in 输入的样本数
%n_out 输出的样本数
fanin=n_in*n_out;
sd=1/sqrt(fanin);
w=sd*randn(n_out,n_in);
b=zeros(n_out,1);
softmax.w=w;
softmax.b=b;
end

