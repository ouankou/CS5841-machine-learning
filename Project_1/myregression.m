function [pred] = myregression(trainX, testX, noutput)

%train the model with cross validation

    data = trainX;
    for cv = 1:10, % random cross validation
        
        [nr nc] = size(trainX);
        w0 = ones([nr 1]);
        data = [w0 trainX];
        
        cvindex = randperm(nr); % randomly permutes indices of data used for cv
    
        trainxX2 = data(cvindex(1:floor(nr*4/5)),1:end-noutput);
        trainxY2 = data(cvindex(1:floor(nr*4/5)),end-noutput+1:end);

        testx = data(cvindex(ceil(nr*4/5):end),1:end-noutput);
        testt = data(cvindex(ceil(nr*4/5):end),end-noutput+1:end);
        
        [nr nc] = size(trainxX2);
        for i = 2:nc
            max_col = max(trainxX2(:,i));
            min_col = min(trainxX2(:,i));
            ave_col = mean(trainxX2(:,i));
            var_col = var(trainxX2(:,i));
            for j = 1:nr
                %trainxX2(j,i) = (trainxX2(j,i)-min_col)/(max_col-min_col);
                trainxX2(j,i) = exp(-(trainxX2(j,i)-ave_col)^2/2*var_col);
            end
        end

        W(:,:,cv) = pinv(transpose(trainxX2)*trainxX2)*transpose(trainxX2)*trainxY2;

        [nr nc] = size(testx);
        for i = 2:nc
            max_col = max(testx(:,i));
            min_col = min(testx(:,i));
            ave_col = mean(testx(:,i));
            var_col = var(testx(:,i));

            for j = 1:nr
                %testx(j,i) = (testx(j,i)-min_col)/(max_col-min_col);
                testx(j,i) = exp(-(testx(j,i)-ave_col)^2/2*var_col);

            end
        end

        pred = testx*W(:,:,cv);
        
        sqerr(cv) = sum((testt(:)-pred(:)).^2);
    end;

%choose the optimal parameters

    [M,I] = min(sqerr);
    W_optimal = W(:,:,I);
    
    
    
%process given testing data

    [nr nc] = size(testX);
    w0 = ones([nr 1]);
    testX2 = [w0 testX];

    [nr nc] = size(testX2);
    for i = 2:nc
        max_col = max(testX2(:,i));
        min_col = min(testX2(:,i));
        ave_col = mean(testX2(:,i));

        var_col = var(testX2(:,i));
        for j = 1:nr
            testX2(j,i) = exp(-(testX2(j,i)-ave_col)^2/2*var_col);

            %testX2(j,i) = (testX2(j,i)-min_col)/(max_col-min_col);
        end
    end
    
    pred = testX2*W_optimal;