function error = testStackAutoencoder( net,samples,labels )
%≤‚ ‘stack autoencoder
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
[pre,pre_labels]=max(a3);
pre_labels=pre_labels';
error=mean(pre_labels~=labels);
end

