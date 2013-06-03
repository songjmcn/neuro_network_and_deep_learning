function active = Tanh()
%tanhÐÍ¼¤»îº¯Êý
active.fun=@Fun;
active.dfun=@DFun;

end
function out=Fun(e)
out=tanh(e);
end
function out=DFun(e)
tmp=Fun(e);
out=1-tmp.*tmp;
end

