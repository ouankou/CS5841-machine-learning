% set random state
rand ('state', sum(100*clock));
randn('state', sum(100*clock));

% set command window format
format compact
format short g

% set some other default values
set(0, 'RecursionLimit', 50);
set(0, 'DefaultFigureWindowStyle', 'normal');
set(0, 'DefaultAxesBox', 'on');
set(0,'DefaultLineLineWidth',4);
set(0,'DefaultLineMarkerSize',12);
set(0,'DefaultTextFontSize',16);
set(0,'DefaultAxesFontSize',16);
set(0,'DefaultAxesFontWeight','bold');
set(0,'DefaultTextFontWeight','bold');
set(0,'DefaultAxesColor','white');
set(0,'DefaultFigureColor','white');
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultUicontrolFontSize', 8);
recycle('off');



k = (1:50)';

%[pred, bl] = myWKNN(Xtrain, Ytrain, Xtest)


% KNN
[nr,nc] = size(Xtrain);
max_cv = 5;
err = zeros([max_cv,1]);
pred = zeros([max_cv,nr-floor(nr*4/5)]);
for cv = 1:max_cv, % random cross validation
    cvindex = randperm(nr); % randomly permutes indices of data used for cv
    
    trainx = Xtrain(cvindex(1:floor(nr*9/10)),:);
    trainy = Ytrain(cvindex(1:floor(nr*9/10)),:);
    Kt = Xtrain(cvindex(floor(nr*9/10)+1:end),:);
    Kt_y = Ytrain(cvindex(floor(nr*9/10)+1:end),:);
    %[pred(cv,:), bk, errs] = myKNN(trainx, trainy, Kt, k);
    %[preds, bl] = myWKNN(trainx, trainy, Kt)
    [preds, bk, errs] = myKNN(trainx, trainy, Kt, k);
    %compare pred and Kt labels
    [pr,~] = size(preds);
    for i = 1:pr
        if Kt_y(i) ~= preds(i)
            err(cv) = err(cv) + 1;
        end
    end;
    %preds
    %err(cv)
end;





% data_x = K3;
% data_y = Y3;
% dataset = 1;
% [nr,nc] = size(data_x);
% max_cv = 1;
% err = zeros([max_cv,1]);
% prediction = zeros([max_cv,nr-floor(nr*4/5)]);
% %alpha = zeros([max_cv,floor(nr*4/5)]);
% %prediction = zeros([max_cv,1]);
% %alpha = zeros([max_cv,1]);
% b = zeros([max_cv,1]);
% for cv = 1:max_cv, % random cross validation
%     cvindex = randperm(nr); % randomly permutes indices of data used for cv
%     
%     trainx = data_x(cvindex(1:floor(nr*9/10)),cvindex(1:floor(nr*9/10)));
%     trainy = data_y(cvindex(1:floor(nr*9/10)),:);
%     Kt = data_x(cvindex(1:floor(nr*9/10)),cvindex(floor(nr*9/10)+1:end));
%     Kt_y = data_y(cvindex(floor(nr*9/10)+1:end));
%     [prediction, alpha, b(cv)] = mySVM(trainx, trainy, Kt, dataset);
%     %[prediction, alpha, b] = mySVM(trainx, trainy, Kt, dataset);
%     %compare pred and Kt labels
%     [pr,~] = size(prediction);
%     for i = 1:pr
%         if Kt_y(i) ~= prediction(i)
%             err(cv) = err(cv) + 1
%         end
%     end;
% end;

