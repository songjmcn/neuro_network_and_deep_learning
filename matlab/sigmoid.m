function active = sigmoid()
%SÐÍ¼¤»îº¯Êý
active.fun=@Fun;
active.dfun=@DFun;
end
function out=Fun(x)
out=1./(1+exp(-x));
end
function out=DFun(x)
tmp=Fun(x);
out=tmp.*(1-tmp);
end
