function dbn = trainDBN( dbn,samples,opt )
%ÑµÁ·DBN
n=length(dbn.rbm);
input=samples;
for i=1:n
    rbm=dbn.rbm{i};
    rbm=trainRBM(rbm,input,opt);
    input=forward_rbm(rbm,input);
    if(i~=n)
        dbn.rbm{i}=rbm;
    end
end
end
function out=forward_rbm(rbm,data)
out=rbm.active.fun(rbm.w*data+repmat(rbm.c,1,size(data,2)));
end

