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

% load all the datasets
train_x = load('train.data');
train_y = load('train.label');
test_x = load('test.data');
test_y = load('test.label');

% initalize parameters and matrices
delta = 0.06;
m = max(train_x(:,2));
docs = zeros(max(train_x(:,1)), m);
label_word = zeros(max(train_y(:,1)), m);

% calculate term probability matrix
[trainx_r, trainx_c] = size(train_x);
for i = 1:trainx_r
    docs(train_x(i,1), train_x(i,2)) = train_x(i,3);
    label_word(train_y(train_x(i,1),1),train_x(i,2)) = label_word(train_y(train_x(i,1),1),train_x(i,2)) + train_x(i,3);
end

for i = 1:max(train_y(:,1))
    c_sum = sum(label_word(i,:));
    for j = 1:m
        label_word(i,j) = (1-delta)*label_word(i,j)/c_sum + delta/m;
    end
end

% calculate proir probability
pck = zeros(1, max(train_y(:,1)));
for i = 1:max(train_x(:,1))
    pck(train_y(i)) = pck(train_y(i)) + 1;
end    
pck = pck/sum(pck);

% testing starts
% initialize the probability matrix with prior probabiliy
test_results = zeros(max(test_x(:,1)), max(train_y(:,1)));
[testx_r, ~] = size(test_x);
for i = 1:max(train_y(:,1))
    test_results(:,i) = log(pck(i));
end

% summation of log probabilities
for i = 1:testx_r
    for j = 1:max(train_y(:,1))
        if test_x(i,2) <= m
            test_results(test_x(i,1),j) = test_results(test_x(i,1),j) + test_x(i,3)*log(label_word(j,test_x(i,2)));
        end
    end
end

% output final prediction, which is the maximum probability among 20 labels
test_labels = zeros(max(test_x(:,1)), 1);
[testx_r, testx_c] = size(test_x);
for i = 1:max(test_x(:,1))
     [prob, label]= max(test_results(i,:));
     test_labels(i) = label;
end

% calculate accuracy of prediction
accu = sum(test_y == test_labels)/max(test_x(:,1));



c = 2
t = 1;
while (test_y(t) ~= c)
    t = t+1;
end

total = 0;
sub_accu = 0;
miss = [];
while (test_y(t) == c)
    total = total+1;
    if test_y(t) == test_labels(t)
        sub_accu = sub_accu + 1;
    else
        miss = [miss test_labels(t)];
    end
    t = t+1;
    if t > max(test_x(:,1))
        break;
    end
end
sub_accu = sub_accu/total
unique(miss)
sort(miss)

