function [C, gamma, model, minCrossValError, errors] = gridSearch(X, Y, k, costs, gammas)
    errors = zeros(length(costs), length(gammas));
    minCrossValError = 10^9;

    for i = 1 : length(costs)
        for j = 1 : length(gammas)
            t = templateSVM('KernelFunction', 'gaussian', 'BoxConstraint', costs(i), 'KernelScale', gammas(j));
            mdl = fitcecoc(X, Y, 'Learners', t);
            cMdl = crossval(mdl, 'KFold', k);
            errors(i, j) = kfoldLoss(cMdl);

            if (errors(i, j) < minCrossValError)
                minCrossValError = errors(i, j);
                C = costs(i);
                gamma = gammas(j);
                model = mdl;
            end
        end
    end
end