function net = trainAotuencoder( net, samples,eta)
%训练autoencoder
[n,m]=size(samples);
bitch_size=500;
loop=100;
%前向传播
n_train=floor(m/bitch_size);
ss=0;
%前向传播
for lp=1:loop
fprintf('开始前向传播，计算隐藏层的输出和\n');
    start_index=1;
    for ii=1:n_train
        end_index=ii*bitch_size;
        this_samples=samples(:,start_index:end_index);
        this_sum=forward_prop(net,this_samples);
        ss=ss+this_sum;
        start_index=end_index+1;
    end
    rho=(1/m)*ss;
    fprintf('开始反向传播，调整各层参数\n');
    start_index=1;
    cost=0;
    for ii=1:n_train
        end_index=ii*bitch_size;
        this_samples=samples(:,start_index:end_index);
        [net,this_cost]=backward_prop(net,this_samples,rho,eta);
        start_index=end_index+1;
        cost=cost+this_cost;
    end
    cost=cost./n_train;
    fprintf('cost %g\n',cost);
end
end
function out=forward_prop(net,samples)
[n,m]=size(samples);
W1=net.hidden.w;
b1=net.hidden.b;
W2=net.out.w;
b2=net.out.b;
z2=W1*samples+repmat(b1,1,m);
a2=sigmoid(z2);
z3=W2*a2+repmat(b2,1,m);
a3=sigmoid(z3);
out=sum(a2,2);
end
function [net,cost]=backward_prop(net,samples,rho,eta)
beta=3;
lambda=1e-4;
sparsityParam=0.01;
[n,m]=size(samples);
W1=net.hidden.w;
b1=net.hidden.b;
W2=net.out.w;
b2=net.out.b;
z2=W1*samples+repmat(b1,1,m);
a2=sigmoid(z2);
z3=W2*a2+repmat(b2,1,m);
a3=sigmoid(z3);
Jcost = (0.5/m)*sum(sum((a3-samples).^2));
Jweight = (1/2)*(sum(sum(W1.^2))+sum(sum(W2.^2)));
a3=sigmoid(z3);
Jsparse = sum(sparsityParam.*log(sparsityParam./rho)+ ...
        (1-sparsityParam).*log((1-sparsityParam)./(1-rho)));
cost = Jcost+lambda*Jweight+beta*Jsparse;
d3=-(samples-a3).*dsigmoid(z3);
sterm=beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));
d2=(W2'*d3+repmat(sterm,1,m)).*dsigmoid(z2);
dw1=(1/m)*(d2*samples')+lambda*W1;
dw2=(1/m)*(d3*a2')+lambda*W2;
db1=(1/m)*sum(d2,2);
db2=(1/m)*sum(d3,2);
net.hidden.w=W1-eta*dw1;
net.hidden.b=b1-eta*db1;
net.out.w=W2-eta*dw2;
net.out.b=b2-eta*db2;
end
