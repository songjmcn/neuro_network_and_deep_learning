function dbn = buildDBN(n_layers)
%初始化DBN网络
%去掉输入层
n=length(n_layers)-1;
for i=1:n
    n_in=n_layers(i);
    n_out=n_layers(i+1);
    dbn.rbm{i}=buildRBM(n_in,n_out,sigmoid());
end
end

