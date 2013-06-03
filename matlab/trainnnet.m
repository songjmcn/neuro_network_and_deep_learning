function nnet = trainnnet( nnet,samples,labels,opt)
%ÑµÁ·Éñ¾­ÍøÂç
%nnet the parameter of the nnet
%samples the need samples
%labels the labels
%opt train parameters have 
%opt.eta opt.batchsize opt.numepoches
eta=opt.eta;
batchsize=opt.batchsize;
numepoches=opt.numopeches;
m=size(samplrs,2);
n_train=floor(m/batchsize);
n_l=length(nnet.w);
for lp=1:numepoches
    for ii=1:n_train
        for l=1:n_l
        this_samples=samples(:,(ii-1)*batchsize+1:ii*batchsize);
        this_labels=labels(:,(ii-1)*batchsize+1:ii*batchsize);
        outs=forward(nnet,this_samples);
        nnet=backward(nnet,this_samples,this_labels,outs,eta);
        end
    end
end

end
function out=forward(nnet,samples)
n_l=length(nnet.w);
input=samples;
m=size(samples,2);
for i=1:n_l
    a=nnet.active{i}.fun(nnet.w{i}*input+repmat(nnet.b{i},1,m));
    input=a;
    out{i}=a;
end
end
function [dw,db]=backward(nnet,samples,labels,outs,eta)
m=size(samples,2);
n_layers=lenfth(nnet.w);
targetOut=zeros(outs{n_layers});
for ii=1:m
    targetOut(labels(ii),ii)=1;
end
dout=-(targetOut-outs{n_layers});
for i=n_layers:-1:1
    dout=dout.*nnet.active{i}.dfun(outs{i});
    dw=(1/m)*dout*outs{i-1};
    db=s(1/m).*sum(dout,2);
    dout=nnet.w{i}'*dout;
    nnet.w{i}=nnet.w{i}-eta.*dw;
    nnet.b{i}=nnet.b{i}-eta.*db;
end
end


