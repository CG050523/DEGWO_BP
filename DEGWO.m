function [Alpha_score,Alpha_pos,Convergence_curve]=DEGWO(SearchAgents_no,Max_iter,lb,ub,dim,fobj)

% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim);
Alpha_score=inf; % 将其更改为 -inf 以解决最大化问题

Beta_pos=zeros(1,dim);
Beta_score=inf; % 将其更改为 -inf 以解决最大化问题

Delta_pos=zeros(1,dim);
Delta_score=inf; % 将其更改为 -inf 以解决最大化问题

% 初始化搜索代理的位置
Positions=initialization(SearchAgents_no,dim,ub,lb);
%% 改进点：利用差分变异产生初始种群
%边界检查
for i=1:size(Positions,1)  
        
       % 返回超出搜索空间边界的搜索代理
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; 
end
%差分变异
for i=1:size(Positions,1)  
    index = randi(SearchAgents_no,[1,3]);%随机选取3只狼
    F = (0.8 -0.2)*rand() + 0.2;%缩放因子范围[0.2,0.8]
    Temp = Positions(index(1),:) + F.*( Positions(index(2),:) - Positions(index(3),:));
     Flag4ub=Temp>ub;
     Flag4lb=Temp<lb;
     Temp=(Temp.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
     if(fobj(Temp)<fobj(Positions(i,:)))
         Positions(i,:) = Temp;
     end
end

Convergence_curve=zeros(1,Max_iter);

l=0;% 循环计数器

% 主循环
while l<Max_iter
    l
    for i=1:size(Positions,1)  
        
       % 返回超出搜索空间边界的搜索代理
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;               
        
        % 计算每个搜索代理的目标函数
        fitness=fobj(Positions(i,:));
        
        % 更新 Alpha, Beta, and Delta
        if fitness<Alpha_score 
            Alpha_score=fitness; % 更新 alpha
            Alpha_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness<Beta_score 
            Beta_score=fitness; % 更新 beta
            Beta_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score 
            Delta_score=fitness; % 更新 delta
            Delta_pos=Positions(i,:);
        end
    end
       
    a=2-l*((2)/Max_iter); % a 线性减少 fron 2 到 0
    
    % 更新搜索代理（包括 omega）的位置
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)     
                       
            r1=rand(); % r1 是 [0,1] 中的随机数
            r2=rand(); % r2 是 [0,1] 中的随机数
            
            A1=2*a*r1-a; % 方程 (3.3)
            C1=2*r2; % 方程 (3.4)
            
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % 方程 (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % 方程 (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % 方程 (3.3)
            C2=2*r2; % 方程 (3.4)
            
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % 方程 (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % 方程 (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % 方程 (3.3)
            C3=2*r2; % 方程 (3.4)
            
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % 方程 (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % 方程 (3.5)-part 3             
            
            Positions(i,j)=(X1+X2+X3)/3;% 方程 (3.7)
            
        end
        %% 改进点：差分变异
        index = randi(SearchAgents_no,[1,3]);%随机选取3只狼
        F = (0.8 -0.2)*rand() + 0.2;%缩放因子范围[0.2,0.8]
        H = Positions(index(1),:) + F.*( Positions(index(2),:) - Positions(index(3),:));
        Flag4ub=H>ub;
        Flag4lb=H<lb;
        H=(H.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        CR = 0.2;%交叉概率
        for j = 1:dim
            if rand<CR
                V(j) = H(j);
            else
                V(j) = Positions(i,j);
            end
        end
        if(fobj(V)<fobj( Positions(i,:)))
            Positions(i,:) = V;
        end
        
        
    end
    l=l+1;    
    Convergence_curve(l)=Alpha_score;
end



