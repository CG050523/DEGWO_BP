%DLH-GWO-BPNN���Ĵ���
clear
clc
close all
tic
global SamIn SamOut HiddenUnitNum InDim OutDim TrainSamNum 
%% ����ѵ������
data = xlsread('2.xlsx');
[data_m,data_n] = size(data);%��ȡ����ά��

P = 80;  %�ٷ�֮P����������ѵ��
Ind = floor(P * data_m / 100);

train_data = data(1:Ind,1:end-1)'; 
train_result = data(1:Ind,end)';
n = length(unique(train_result));
%train_result = full(ind2vec(train_result,n));

test_data = data(Ind+1:end,1:end-1)';% ����ѵ���õ��������Ԥ��
test_result = data(Ind+1:end,end)';
%test_result = full(ind2vec(test_result,n));

%% ��ʼ������
[InDim,TrainSamNum] = size(train_data);% ѧϰ��������
[OutDim,TrainSamNum] = size(train_result);
HiddenUnitNum = 6;                     % ��������Ԫ����


[SamIn,PS_i] = mapminmax(train_data,0,1);    % ��һ��
[SamOut,PS_o] = mapminmax(train_result,0,1);

W1 = HiddenUnitNum*InDim;      % ��ʼ���������������֮���Ȩֵ
B1 = HiddenUnitNum;          % ��ʼ���������������֮�����ֵ
W2 = OutDim*HiddenUnitNum;     % ��ʼ���������������֮���Ȩֵ
B2 = OutDim;                % ��ʼ���������������֮�����ֵ

L = W1+B1+W2+B2;        %����ά��
%%�Ż��������趨
dim=L; % �Ż��Ĳ��� number of your variables
for j=1:L
lb(1,j)=-3.55; % ����ȡֵ�½�3.55   5.55
ub(1,j)=3.55;
end% ����ȡֵ�Ͻ�
Boundary_no= size(ub,2);


%%GWO�㷨��ʼ��
SearchAgents_no=30; % ��Ⱥ������Number of search agents   50
Max_iteration=100; % ������������Maximum numbef of iterations   500

lu = [lb .* ones(1, dim); ub .* ones(1, dim)];

% initialize alpha, beta, and delta_pos
Alpha_score=inf; % ��ʼ��Alpha�ǵ�Ŀ�꺯��ֵ��change this to -inf for maximization problems
Alpha_pos=zeros(1,dim); % ��ʼ��Alpha�ǵ�λ��
Beta_pos=zeros(1,dim); % ��ʼ��Beta�ǵ�λ��
Beta_score=inf; % ��ʼ��Beta�ǵ�Ŀ�꺯��ֵ��change this to -inf for maximization problems

Delta_pos=zeros(1,dim); % ��ʼ��Delta�ǵ�λ��
Delta_score=inf; % ��ʼ��Delta�ǵ�Ŀ�꺯��ֵ��change this to -inf for maximization problems

%Initialize the positions of search agents
Positions=initialization(SearchAgents_no,dim,ub,lb);
Positions = boundConstraint (Positions, Positions, lu);

% ����ÿͷ�ǵ���Ӧ��
for i=1:size(Positions,1)
    Fit(i) = f(Positions(i,:));
end

% Personal best fitness and position obtained by each wolf   
% ÿֻ�ǻ�õĸ��������Ӧ�Ⱥ�λ��
pBestScore = Fit;
pBest = Positions;
neighbor = zeros(SearchAgents_no,SearchAgents_no);
Convergence_curve=zeros(1,Max_iteration);

l=0; % Loop counterѭ��������

    % Main loop��ѭ��
    while l<Max_iteration  % �Ե�������ѭ��
        l
        for i=1:size(Positions,1)  % ����ÿ����
            
            % ������Ӧ�Ⱥ���ֵ
            fitness=Fit(i);
            
            % Update Alpha, Beta, and Delta
            if fitness<Alpha_score % ���Ŀ�꺯��ֵС��Alpha�ǵ�Ŀ�꺯��ֵ
                Alpha_score=fitness; % ��Alpha�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ��Update alpha
                Alpha_pos=Positions(i,:); % ͬʱ��Alpha�ǵ�λ�ø���Ϊ����λ��
            end
            
            if fitness>Alpha_score && fitness<Beta_score % ���Ŀ�꺯��ֵ������Alpha�Ǻ�Beta�ǵ�Ŀ�꺯��ֵ֮��
                Beta_score=fitness; % ��Beta�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ��Update beta
                Beta_pos=Positions(i,:); % ͬʱ����Beta�ǵ�λ��
            end
            
            if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score  % ���Ŀ�꺯��ֵ������Beta�Ǻ�Delta�ǵ�Ŀ�꺯��ֵ֮��
                Delta_score=fitness; % ��Delta�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ��Update delta
                Delta_pos=Positions(i,:); % ͬʱ����Delta�ǵ�λ��
            end
        end
        
        a=2-l*((2)/Max_iteration); % ��ÿһ�ε�����������Ӧ��aֵ��a decreases linearly fron 2 to 0
        
        
       
        
        % Update the Position of search agents including omegas
        for i=1:size(Positions,1) % ����ÿ����
            for j=1:size(Positions,2) % ����ÿ��ά��
                
                % ��Χ���λ�ø���
                
                r1=rand(); % r1 is a random number in [0,1]
                r2=rand(); % r2 is a random number in [0,1]
                
                A1=2*a*r1-a; % ����ϵ��A��Equation (3.3)
                C1=2*r2; % ����ϵ��C��Equation (3.4)
                
                % Alpha��λ�ø���
                D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
                X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                
                r1=rand();
                r2=rand();
                
                A2=2*a*r1-a; % ����ϵ��A��Equation (3.3)
                C2=2*r2; % ����ϵ��C��Equation (3.4)
                
                % Beta��λ�ø���
                D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
                X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2
                
                r1=rand();
                r2=rand();
                
                A3=2*a*r1-a; % ����ϵ��A��Equation (3.3)
                C3=2*r2; % ����ϵ��C��Equation (3.4)
                
                % Delta��λ�ø���
                D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
                X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3
                
                % λ�ø���
                Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
                X_GWO(i,j)=(X1+X2+X3)/3;
                
            end
            X_GWO(i,:) = boundConstraint(X_GWO(i,:), Positions(i,:), lu);
            Fit_GWO(i) = f(X_GWO(i,:));
        end
        % Calculate the candiadate position Xi-DLH
        radius = pdist2(Positions, X_GWO, 'euclidean');    %����X������һ����������Y������һ���������ľ���,Ĭ�ϲ���ŷ�Ͼ��빫ʽ �������ķ���ֵΪ����D��D�Ǿ���һ�У�(m*(m-1)/2)�е���������    % Equation (10)
        dist_Position = squareform(pdist(Positions));  
        % ��һ�У���һ�У���������ɷ���
        r1 = randperm(SearchAgents_no,SearchAgents_no);   %ǰSearchAgents_no������ѡSearchAgents_no��
        
        for t=1:SearchAgents_no
            neighbor(t,:) = (dist_Position(t,:)<=radius(t,t));
            [~,Idx] = find(neighbor(t,:)==1);                   % Equation (11)
            random_Idx_neighbor = randi(size(Idx,2),1,dim);
            
            for d=1:dim
                X_DLH(t,d) = Positions(t,d) + rand .*(Positions(Idx(random_Idx_neighbor(d)),d)...
                    - Positions(r1(t),d));                      % Equation (12)
            end
            X_DLH(t,:) = boundConstraint(X_DLH(t,:), Positions(t,:), lu);
            Fit_DLH(t) = f(X_DLH(t,:));
        end
        
        % Selection
        tmp = Fit_GWO < Fit_DLH;                                % Equation (13)
        tmp_rep = repmat(tmp',1,dim);
        
        tmpFit = tmp .* Fit_GWO + (1-tmp) .* Fit_DLH;
        tmpPositions = tmp_rep .* X_GWO + (1-tmp_rep) .* X_DLH;
        
        % Updating
        tmp = pBestScore <= tmpFit;                             % Equation (13)
        tmp_rep = repmat(tmp',1,dim);
        
        pBestScore = tmp .* pBestScore + (1-tmp) .* tmpFit;
        pBest = tmp_rep .* pBest + (1-tmp_rep) .* tmpPositions;
        
        Fit = pBestScore;
        Positions = pBest;
        
        
        l=l+1;
        neighbor = zeros(SearchAgents_no,SearchAgents_no);
        Convergence_curve(l)=Alpha_score;
        Alpha_score
    end


x=Alpha_pos;
x
%%
% % x = gb;
% W1 = x(1:HiddenUnitNum*InDim);
% L1 = length(W1);
% W1 = reshape(W1,[HiddenUnitNum, InDim]);
% B1 = x(L1+1:L1+HiddenUnitNum)';
% L2 = L1 + length(B1);
% W2 = x(L2+1:L2+OutDim*HiddenUnitNum);
% L3 = L2 + length(W2);
% W2 = reshape(W2,[OutDim, HiddenUnitNum]);
% B2 = x(L3+1:L3+OutDim)';
% HiddenOut = tansig(W1 * SamIn + repmat(B1, 1, TrainSamNum));   % �������������tansig
% NetworkOut = W2 * HiddenOut + repmat(B2, 1, TrainSamNum);      % ������������
% temp1 = softmax(NetworkOut);
% Error1 = SamOut - temp1;       % ʵ��������������֮��
%����Ȩֵ��ֵ
inputnum=InDim;
hiddennum=HiddenUnitNum;
outputnum=OutDim;
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
inputn=SamIn;
outputn=SamOut;
net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;
%BP�����繹��
net=newff(inputn,outputn,hiddennum);
net.trainParam.epochs=20;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;
net.trainParam.show=100;
net.trainParam.showWindow=0;

%BP������ѵ��
net=train(net,inputn,outputn);

%����ѵ��
temp1=sim(net,inputn);

Error1=sum(abs(temp1-outputn));



Forcast_data = mapminmax('reverse',temp1,PS_o);

plot(Forcast_data, 'b*-', 'LineWidth', 2); % ����Ԥ�����ݣ���ɫʵ��
hold on
plot(train_result, 'r--', 'LineWidth', 1.5); % ����ѵ���������ɫ����
legend('��ʵֵ', 'Ԥ��ֵ'); % ���ͼ��
xlabel('��������'); % ���� x ���ǩ
ylabel('��ֵ'); % ���� y ���ǩ
title('ѵ�����������ʵֵ��Ԥ��ֵ�Ա�'); % ��ӱ���

% [val1, index1] = max(Forcast_data,[],1);
% [val11, index11] = max(train_result,[],1);
% p1 = 0;
% for i=1:size(train_result,2)
%     if index1(i) == index11(i)
%         p1 = p1 + 1;
%     end
% end
% predict1 = p1 / size(train_result,2)

[OutDim,ForcastSamNum] = size(test_result);
SamIn_test= mapminmax('apply',test_data,PS_i); % ԭʼ�����ԣ�������������ʼ��
% HiddenOut_test = tansig(W1 * SamIn_test + repmat(B1, 1, ForcastSamNum));  % ���������Ԥ����
% NetworkOut_test = W2 * HiddenOut_test + repmat(B2, 1, ForcastSamNum);          % ��������Ԥ����
% temp2 = softmax(NetworkOut_test);
% Forcast_data_test = mapminmax('reverse',temp2,PS_o);
temp2=sim(net,SamIn_test);
Forcast_data_test = mapminmax('reverse',temp2,PS_o);


figure
plot(Forcast_data_test, 'b*-', 'LineWidth', 2); % ����Ԥ�����ݣ���ɫʵ��
hold on
plot(test_result, 'r--', 'LineWidth', 1.5); % ����ѵ���������ɫ����
legend('Ԥ��ֵ', '��ʵֵ'); % ���ͼ��
xlabel('��������'); % ���� x ���ǩ
ylabel('��ֵ'); % ���� y ���ǩ
title('���Լ��������ʵֵ��Ԥ��ֵ�Ա�'); % ��ӱ���


% [val2, index2] = max(Forcast_data_test,[],1);
% [val22, index22] = max(test_result,[],1);
% p2 = 0;
% for i=1:size(test_result,2)
%     if index2(i) == index22(i)
%         p2 = p2 + 1;
%     end
% end
% predict2 = p2 / size(test_result,2)
% toc

%% ���ƽ��
figure

plot(Convergence_curve,'r')
xlabel('��������')
ylabel('��Ӧ��')
title('��������')

% ����ѵ�������

figure
% ����ѵ�����������ʵֵ��Ԥ��ֵ�Ա�ͼ
subplot(1,2,1);
plot(Forcast_data-train_result, 'b*-'); % ����ѵ���������ͼ
xlabel('��������'); % ���� x ���ǩ
ylabel('���ֵ'); % ���� y ���ǩ
title('ѵ�������'); % ��ӱ���

% ���Ʋ��Լ��������ʵֵ��Ԥ��ֵ�Ա�ͼ
subplot(1,2,2);
plot(Forcast_data_test-test_result, 'b*-'); % ���Ʋ����������ͼ
xlabel('��������'); % ���� x ���ǩ
ylabel('���ֵ'); % ���� y ���ǩ
title('���Լ����'); % ��ӱ���

toc