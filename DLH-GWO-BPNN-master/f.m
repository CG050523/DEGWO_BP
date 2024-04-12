function y = f( x )
global SamIn SamOut HiddenUnitNum InDim OutDim TrainSamNum



inputnum=InDim;
hiddennum=HiddenUnitNum;
outputnum=OutDim;
%提取 BP神经网络初始权值和阈值，x为个体
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
inputn=SamIn;
outputn=SamOut;
 
%网络权值赋值
net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;
%BP神经网络构建
net=newff(inputn,outputn,hiddennum);
net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;
net.trainParam.show=100;
net.trainParam.showWindow=0;

%BP神经网络训练
net=train(net,inputn,outputn);

%网络训练
an=sim(net,inputn);

y=sum(abs(an-outputn));



end

