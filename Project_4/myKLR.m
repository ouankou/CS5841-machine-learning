

function [prediction] = myKLR(Xtrain, Ytrain, Xtest)
    
    C = 1; %Parameter C doesn't affect the result much.
    lambda = 0.01;
    K = Xtrain*Xtrain'; %Linear kernel calculation
    K = (1 + Xtrain*Xtrain')^2; %Polynomial kernel calculation
    % calculate alpha with fmincon
    [nr, nc] = size(Xtrain);
    Aeq = Ytrain';
    beq = 0;
    lb = [];
    ub = [];
    a0 = ones(nr, 1)*0.5;
    options = optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',1e6);
    a = fmincon(@(a)calc_func(a, K, Ytrain, lambda, C), a0, [], [], Aeq, beq, lb, ub, [], options);
    % calculate w parameter with alpha
    w = 0;
    for i = 1:nr
        w = w + a(i)*Ytrain(i)*Xtrain(i,:)'/lambda;
    end
    % calculate parameter b with fminunc
    b = fminunc(@(b)calc_b(b, Xtrain, w, Ytrain), -0.5);
    % predict training set
    pred = zeros(nr, 1);
    for i = 1:nr
        pred(i) = 1/(1+exp(-((w'*Xtrain(i,:)'+b))));
        if pred(i) > 0.5
            pred(i) = 1;
        else
            pred(i) = -1;
        end

    end
    accu = sum(pred == Ytrain)/nr
    
    % predict testing set
    [nrt, ~] = size(Xtest);
    prediction = zeros(nrt, 1);
    for i = 1:nrt
        prediction(i) = 1/(1+exp(-(w'*Xtest(i,:)'+b)));
        if prediction(i) > 0.5
            prediction(i) = 1;
        else
            prediction(i) = -1;
        end
    end
        
end

function [fun] = calc_func(a, K, Ytrain, lambda, C)
    
    fun = 0;
    [nr, ~] = size(Ytrain);
    for i = 1:nr
        for j = 1:nr
            fun = fun - a(i)*a(j)*Ytrain(i)*Ytrain(j)*K(i,j)/(2*lambda);
        end
        fun = fun - a(i)*log(a(i)) - (C-a(i))*log(C-a(i));
    end    
end

function [fun_b] = calc_b(b, Xtrain, w, Ytrain)
    fun_b = 0;
    [nr, ~] = size(Xtrain);
    for i = 1:nr
        fun_b = fun_b + log(1+exp(-Ytrain(i)*(w'*Xtrain(i,:)'+b)));
    end

end




