function [pred] = myregression(trainX, testX, noutput)

    [nr nc] = size(trainX);
    w0 = ones([nr 1]);
    trainX2 = [w0 trainX];
    
    [nr nc] = size(testX);
    w0 = ones([nr 1]);
    testX2 = [w0 testX];

    trainxX2 = trainX2(1:end,1:end-noutput);
    trainxY2 = trainX2(1:end,end-noutput+1:end);

    W = inv(transpose(trainxX2)*trainxX2)*transpose(trainxX2)*trainxY2;
    pred = testX2*W;