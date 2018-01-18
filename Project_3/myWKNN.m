

function [prediction, bestlambda] = myWKNN(X, Y, Xtest)
    
    % the best lambda is 1.2.

    [nr, nc] = size(X);
    maxl = 100; % iterations of testing lambda
    errors = zeros([maxl, 1]);
    
    for i = 1:maxl
        t = 0.6+0.05*i; % test lambda from 0.6 to 5.6. According to tests not shown in the submitted codes, larger lambda till 100 will lead to more errors.
        for p = 1:nr
            % Calculate distances
            distance = zeros([nr 1]);
            for q = 1:nr
                for j = 1:nc
                    distance(q) = distance(q) + (X(p,j) - X(q,j))^2;
                end
                distance(q) = sqrt(distance(q));

            end
            [~, nearest] = sort(distance, 'ascend');
            % Calculate weights based on distances.
            weight = zeros([max(Y),1]);
            for q = 2:nr
                weight(Y(nearest(q))) = weight(Y(nearest(q))) + exp(-t*distance(nearest(q)));
            end
            [~, pred_p] = max(weight);
            if pred_p ~= Y(p)
                errors(i) = errors(i) + 1;
            end
        end
    end
    
    [~, bestlambda] = min(errors);
    bestlambda = bestlambda*0.05+0.6; % recover best lambda from iteration index.
        
    %prediction on testing data set
    [nrt, nct] = size(Xtest);
    distance = zeros([nr 1]);
    prediction = zeros([nrt 1]);
    
    for p = 1:nrt
        % Calculate distances.
        for q = 1:nr
            for j = 1:nc
                distance(q) = distance(q) + (Xtest(p,j) - X(q,j))^2;
            end
            distance(q) = sqrt(distance(q));
        end
        [~, nearest] = sort(distance, 'ascend');
        % Calculate weights based on distances.
        weight = zeros([max(Y),1]);
        for q = 2:nr
            weight(Y(nearest(q))) = weight(Y(nearest(q))) + exp(-bestlambda*distance(nearest(q)));
        end
        [~, prediction(p)] = max(weight);
    end
    
end

