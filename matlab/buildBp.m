function net = buildBp(n_in,n_hidden,n_out)
%初始化bp网络的参数
fanin=n_in*n_hidden;
r=sqrt(6)/sqrt(fanin);
w1=r.*rand(n_hidden,n_in)*2-r;
b1=zeros(n_hidden,1);
net.hidden.w=w1;
net.hidden.b=b1;
fanin=n_hidden*n_out;
r=sqrt(r)/sqrt(fanin);
w2=r*rand(n_out,n_hidden)*2-r;
b2=zeros(n_out,1);
net.out.w=w2;
net.out.b=b2;
end

