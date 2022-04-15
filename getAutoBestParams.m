function model = getAutoBestParams(X, Y)
    t = templateSVM('KernelFunction', 'gaussian');
    model = fitcecoc(X, Y, 'Learners', t, 'OptimizeHyperparameters','auto');
end