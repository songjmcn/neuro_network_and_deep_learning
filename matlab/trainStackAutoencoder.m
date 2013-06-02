function net = trainStackAutoencoder( samples,labels,loop )
%训练栈式自编码机
[n,m]=size(samples);
n_out=max(labels);
hidden1=200;
hidden2=200;
net1=buildAutoencoder(n,hidden1);
eta=0.1;
ae_loop=400;
bitch_size=2000;
fprintf('start train autoencoder 1\n');
for lp=1:ae_loop
    net1=trainAotuencoder(net1,samples,eta);
end
net2=buildAutoencoder(hidden1,hidden2);
tmp_samples=net1.hidden.w*samples+repmat(net1.hidden.b,1,m);
fprintf('start train autoencoder 2\n');
for lp=1:400
    net2=trainAotuencoder(net2,tmp_samples,eta);
end
net2=trainAotuencoder(net2,tmp_samples,eta);
net.hidden1.w=net1.hidden.w;
net.hidden1.b=net1.hidden.b;
net.hidden2.w=net2.hidden.w;
net.hidden2.b=net2.hidden.b;
net.classifier.w=rand(n_out,hidden2);
net.classifier.b=zeros(n_out,1);
n_train=floor(m/bitch_size);
eta=0.05;
for ll=1:loop
    start_index=1;
    for i=1:n_train
        end_index=i*bitch_size;
        this_train=samples(:,start_index:end_index);
        this_labels=labels(start_index:end_index);
        net=trainMlp(net,this_train,this_labels,eta);
        start_index=end_index+1;
    end
    error=testStackAutoencoder(net,samples,labels);
    fprintf('第%g训练，错误率为%g%%\n',ll,error*100);
end
end
function net=trainMlp(net,samples,labels,eta)
[n,m]=size(samples);
W1=net.hidden1.w;
b1=net.hidden1.b;
W2=net.hidden2.w;
b2=net.hidden2.b;
W3=net.classifier.w;
b3=net.classifier.b;
z1=W1*samples+repmat(b1,1,m);
a1=sigmoid(z1);
z2=W2*a1+repmat(b2,1,m);
a2=sigmoid(z2);
z3=W3*a2+repmat(b3,1,m);
a3=Softmax(z3);
targetOut=zeros(size(a3));
for ii=1:m
    targetOut(labels(ii),ii)=1;
end
d3=-(targetOut-a3);
dw3=(1/m).*d3*a3';
db3=(1/m).*sum(d3,2);
net.classifier.w=W3-eta*dw3;
net.classifier.b=b3-eta*db3;
d2=(W3'*d3).*dsigmoid(z2);
dw2=(1/m).*d2*a2';
db2=(1/m).*sum(d2,2);
net.hidden2.w=W2-eta*dw2;
net.hidden2.b=b2-eta*db2;
d1=(W2'*d2).*dsigmoid(z1);
dw1=(1/m).*d1*a1';
db1=(1/m).*sum(d1,2);
net.hidden1.w=W1-eta*dw1;
net.hidden1.b=b1-eta*db1;
end

