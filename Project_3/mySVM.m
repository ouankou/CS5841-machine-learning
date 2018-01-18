

function [prediction, alpha, b] = mySVM(K, Y, Kt, dataset)

% Apply 10-fold cross validation to test C values from 1 to 50.
% The result varies based on different partitions of training set and
% testing set. However it does make no significant difference since the
% error is always between 0 to 5 no matter what values C is.
%
% Therefore C is set to 1 constantly.
% The commented codes shown below can are used to automatically generate the
% best C value to apply in the following training and testing purpose.

%     if dataset == 1 || dataset == 2 || dataset == 3
%         max_c = 50;
%         data_x = K;
%         data_y = Y;
%         [nr,~] = size(data_x);
%         max_cv = 10;
%         err = zeros([max_c,1]);
%         for cs = 1:max_c
%             for cv = 1:max_cv
%                 cvindex = randperm(nr); % randomly permutes indices of data used for cv
%                 trainx = data_x(cvindex(1:floor(nr*8/9)),cvindex(1:floor(nr*8/9)));
%                 trainy = data_y(cvindex(1:floor(nr*8/9)),:);
%                 testx = data_x(cvindex(1:floor(nr*8/9)),cvindex(floor(nr*8/9)+1:end));
%                 testy = data_y(cvindex(floor(nr*8/9)+1:end));
% 
%                 %cross validation
%                 
%                 [n,~] = size(trainx);
%                 H = zeros([n,n]);
%                 f = -ones([n,1]);
%                 Aeq = trainy';
%                 beq = 0;
%                 lb = zeros([n,1]);
%                 ub = ones([n,1])*cs;
% 
%                 for i = 1:n
%                     for j = 1:i
%                         if i == j
%                             H(i,j) = 2*trainy(i)*trainy(j)*trainx(i,j);
%                         else
%                             H(i,j) = trainy(i)*trainy(j)*trainx(i,j);
%                             H(j,i) = H(i,j);
%                         end
%                     end
%                 end
% 
%                 options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');
% 
%                 [alpha,fval,exitflag,output,lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);
% 
%                 b = 0;
%                 for j = 1:n
%                     for i = 1:n
%                         b = b - alpha(i)*trainy(i)*trainx(i,j);
%                     end
%                     b = b + trainy(j);
%                 end
%                 b = b/n;
% 
%                 %prediction on testing data set
%                 [~,ntest] = size(testx);
%                 pred = zeros([ntest 1]);
% 
%                 for j = 1:ntest
%                     for i = 1:n
%                         pred(j) = pred(j) + alpha(i)*trainy(i)*testx(i,j);
%                     end
%                     pred(j) = sign(pred(j)+b);
%                 end
% 
%                 [pr,~] = size(pred);
%                 for i = 1:pr
%                     if testy(i) ~= pred(i)
%                         err(cs) = err(cs) + 1;
%                     end
%                 end;
%             end
%             err(cs) = err(cs)/max_cv;
%         end
%         [min_err,bestc] = min(err);
%     else
%         bestc = 1;
%     end
% 
%     C = bestc;
    
    C = 1; % set C to 1 constantly, the reason is explained at the beginning. Comment this line if best C is needed to be found.
    
    % Initialize all variables needed for quadratic programming.
    [n,~] = size(K);
    H = zeros([n,n]);
    f = -ones([n,1]);
    Aeq = Y';
    beq = 0;
    lb = zeros([n,1]);
    ub = ones([n,1])*C;
    
    % Calculate Hessian matrix
    for i = 1:n
        for j = 1:i
            if i == j
                H(i,j) = 2*Y(i)*Y(j)*K(i,j);
            else
                H(i,j) = Y(i)*Y(j)*K(i,j);
                H(j,i) = H(i,j);
            end
        end
    end
    
    % Solve alpha with quadratic programming
    options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');

    [alpha,fval,exitflag,output,lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);
    
    % Calcualte b with given formula.
    b = 0;
    for j = 1:n
        for i = 1:n
            b = b - alpha(i)*Y(i)*K(i,j);
        end
        b = b + Y(j);
    end
    b = b/n;
        
    %prediction on testing data set with given formula.
    [~,ntest] = size(Kt);
     prediction = zeros([ntest 1]);
    
    for j = 1:ntest
        for i = 1:n
            prediction(j) = prediction(j) + alpha(i)*Y(i)*Kt(i,j);
        end
        prediction(j) = sign(prediction(j)+b);
    end
    
end

