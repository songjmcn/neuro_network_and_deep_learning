function net = buildAutoencoder( n_in,n_hidden)
%初始化autoencoder网络参数
net.hidden.w=0.005.*rand(n_hidden,n_in);
net.hidden.b=zeros(n_hidden,1);
net.out.w=0.005*rand(n_in,n_hidden);
net.out.b=zeros(n_in,1);
end

