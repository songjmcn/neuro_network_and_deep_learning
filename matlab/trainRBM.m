function rbm = trainRBM( rbm,samples,opt)
%训练RBM
%rbm 玻尔兹曼机的初始化参数
%samples 样本数据
%opt 训练参数
w=rbm.w;
b=rbm.b;
c=rbm.c;
eta=opt.eta;
active_fun=rbm.active;
numepoches=opt.numepoches;
batchsize=opt.batchsize;
m=size(samples,2);
samples=samples';
numbatches=floor(m/batchsize);
for lop=1:numepoches
    start_index=1;
    for i=1:numbatches
        end_index=i*batchsize;
        this_data=samples(start_index:end_index,:);
        v1=this_data;
        tmp=repmat(c',batchsize,1)+v1*w';
        h1=double(active_fun.fun(tmp)>rand(size(tmp)));
        tmp=repmat(b',batchsize,1)+h1*w;
        v2=double(active_fun.fun(tmp)>rand(size(tmp)));
        tmp=repmat(c',batchsize,1)+v2*w';
        h2=double(active_fun.fun(tmp)>rand(size(tmp)));
        c1=h1'*v1;
        c2=h2'*v2;
        dw=eta*(c1-c2);
        db=eta*sum(v1-v2)';
        dc=eta*sum(h1-h2)';
        w=w+dw;
        b=b+db;
        c=c+dc;
        start_index=end_index+1;
    end
end
rbm.w=w;
rbm.b=b;
rbm.c=c;
end

