function [pred] = myregression(trainX, testX, noutput)

    
%     data = trainX;
%     for cv = 1:10, % random cross validation
%         
%         [nr nc] = size(trainX);
%         w0 = ones([nr 1]);
%         data = [w0 trainX];
%         
%         cvindex = randperm(nr); % randomly permutes indices of data used for cv
%     
%         trainxX2 = data(cvindex(1:floor(nr*4/5)),1:end-noutput);
%         trainxY2 = data(cvindex(1:floor(nr*4/5)),end-noutput+1:end);
% 
%         testx = data(cvindex(ceil(nr*4/5):end),1:end-noutput);
%         testt = data(cvindex(ceil(nr*4/5):end),end-noutput+1:end);
%         
%         [nr nc] = size(trainxX2);
%         for i = 2:nc
%             max_col = max(trainxX2(:,i));
%             min_col = min(trainxX2(:,i));
%             for j = 1:nr
%                 trainxX2(j,i) = (trainxX2(j,i)-min_col)/(max_col-min_col);
%             end
%         end
% 
%         W(:,:,cv) = inv(transpose(trainxX2)*trainxX2)*transpose(trainxX2)*trainxY2;
% 
%         [nr nc] = size(testx);
%         for i = 2:nc
%             max_col = max(testx(:,i));
%             min_col = min(testx(:,i));
%             for j = 1:nr
%                 testx(j,i) = (testx(j,i)-min_col)/(max_col-min_col);
%             end
%         end
% 
%         pred = testx*W(:,:,cv);
%         
%         sqerr(cv) = sum((testt(:)-pred(:)).^2);
%     end;
%     
%     [M,I] = min(sqerr);
%     
%     W_optimal = W(:,:,I);
    
    

    [nr nc] = size(trainX);
    w0 = ones([nr 1]);
    trainX2 = [w0 trainX];
    

    
    trainxX2 = trainX2(1:end,1:end-noutput);
    trainxY2 = trainX2(1:end,end-noutput+1:end);

    [nr nc] = size(trainxX2);
    for i = 2:nc
        max_col = max(trainxX2(:,i));
        min_col = min(trainxX2(:,i));
        ave_col = mean(trainxX2(:,i));
        std_dev = std(trainxX2(:,i));
        for j = 1:nr
            %trainxX2(j,i) = (trainxX2(j,i)-ave_col)/std_dev;
            trainxX2(j,i) = (trainxX2(j,i)-min_col)/(max_col-min_col);
        end
    end
    
    %W = transpose(trainxX2)*trainxX2\(transpose(trainxX2)*trainxY2);
    W_optimal = inv(transpose(trainxX2)*trainxX2)*transpose(trainxX2)*trainxY2;





    [nr nc] = size(testX);
    w0 = ones([nr 1]);
    testX2 = [w0 testX];

    [nr nc] = size(testX2);
    for i = 2:nc
        max_col = max(testX2(:,i));
        min_col = min(testX2(:,i));
        ave_col = mean(testX2(:,i));
        std_dev = std(testX2(:,i));
        for j = 1:nr
            %trainxX2(j,i) = (trainxX2(j,i)-ave_col)/std_dev;
            testX2(j,i) = (testX2(j,i)-min_col)/(max_col-min_col);
        end
    end
    
    pred = testX2*W_optimal;