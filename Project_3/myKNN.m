

function [prediction, bestk, errors] = myKNN(X, Y, Xtest, k)
    
    % the best K is 27, which brings the least error 98.

    [nr, nc] = size(X);
    distance = zeros([nr 1]);
    [kr, ~] = size(k);
    errors = zeros([kr, 1]);
    
    for i = 1:kr % test different k
        for p = 1:nr
            for q = 1:nr
                for j = 1:nc
                    distance(q) = distance(q) + (X(p,j) - X(q,j))^2;
                end
                distance(q) = sqrt(distance(q));
            end
            [~, nearest] = sort(distance, 'ascend'); % sort all distances, the minimum is the checking point itself.
            C = zeros([max(Y),1]);
            for q = 2:k(i)+1 % find k nearest neighbours
                C(Y(nearest(q))) = C(Y(nearest(q))) + 1;
            end
            [~, pred_p] = max(C);
            if pred_p ~= Y(p)
                errors(i) = errors(i) + 1;
            end
        end
    end
    
    [~, bestk] = min(errors); % the index of minimum error is best k
    bestk = k(bestk);
    
    %prediction on testing data set
    [nrt, nct] = size(Xtest);
    distance = zeros([nr 1]);
    prediction = zeros([nrt 1]);
    
%     for p = 1:nr
%         for q = 1:nr
%             for j = 1:nc
%                 distance(q) = distance(q) + (Xtest(p,j) - Xtest(q,j))^2;
%             end
%             distance(q) = sqrt(distance(q));
%         end
%         [~, nearest] = sort(distance, 'ascend'); % sort all distances, the minimum is the checking point itself.
%         C = zeros([max(Y),1]);
%         for q = 2:bestk+1 % find k nearest neighbours
%             C(Y(nearest(q))) = C(Y(nearest(q))) + 1;
%         end
%         [~, prediction(p)] = max(C);
%     end

    for p = 1:nrt % add cv to startup.m to validate
        for q = 1:nr
            for j = 1:nc
                distance(q) = distance(q) + (Xtest(p,j) - X(q,j))^2;
            end
            distance(q) = sqrt(distance(q));
        end
        [~, nearest] = sort(distance, 'ascend'); % sort all distances, the minimum is the checking point itself.
        C = zeros([max(Y),1]);
        for q = 2:bestk+1 % find k nearest neighbours
            C(Y(nearest(q))) = C(Y(nearest(q))) + 1;
        end
        [~, prediction(p)] = max(C);
    end
    
end

