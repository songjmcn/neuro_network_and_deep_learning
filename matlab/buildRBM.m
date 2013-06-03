function rbm = buildRBM( n_in,n_hidden,active)
%创建RBM，并初始化参数
rbm.w=0.1*randn(n_hidden,n_in);
rbm.b=zeros(n_in,1);
rbm.c=zeros(n_hidden,1);
rbm.active=active;
end

