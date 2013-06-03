function active = BipolarSigmoid()
%Ë«S¼¤»îº¯Êý
active.fun=@Fun;
active.dfun=@DFun;
end
function out=Fun(x)
out=(2./(1+exp(-x)))-1;
end
function out=DFun(x)
tmp=exp(-x);
out=(2.*tmp)/((1+tmp).^2);
end

