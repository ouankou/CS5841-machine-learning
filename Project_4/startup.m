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




Xtrain = importdata('heartstatlog_trainSet.txt');
Ytrain = importdata('heartstatlog_trainLabels.txt');
Xtest = importdata('heartstatlog_testSet.txt');
Ytest = importdata('heartstatlog_testLabels.txt');

[nr, nc] = size(Ytrain);
[nrt, nct] = size(Ytest);
for i = 1:nr
    if Ytrain(i) == 2
        Ytrain(i) = -1;
    end
end
for i = 1:nrt
    if Ytest(i) == 2
        Ytest(i) = -1;
    end
end

Xtrain = normc(Xtrain);
Xtest = normc(Xtest);


% [pred] = myKLR(Xtrain, Ytrain, Xtest);
% accu = sum(pred == Ytest)/nrt


% Cross validation
[nr,nc] = size(Xtrain);
max_cv = 5;
accus = zeros([max_cv,1]);
pred = zeros([max_cv,nr-floor(nr*4/5)]);
for cv = 1:max_cv, % random cross validation
    cvindex = randperm(nr); % randomly permutes indices of data used for cv
    
    trainx = Xtrain(cvindex(1:floor(nr*4/5)),:);
    trainy = Ytrain(cvindex(1:floor(nr*4/5)),:);
    Kt = Xtrain(cvindex(floor(nr*4/5)+1:end),:);
    Kt_y = Ytrain(cvindex(floor(nr*4/5)+1:end),:);
    pred = myKLR(trainx, trainy, Kt);
    [nrt, ~] = size(Kt_y);
    accus(cv) = sum(pred == Kt_y)/nrt;
end;
mean(accus)







