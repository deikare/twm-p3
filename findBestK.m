function [k, fig] = findBestK(kVector, model)
    k = kVector(end);
    minKfoldLoss = 10^9;
    
    result = [kVector', zeros(length(kVector), 1)];
    
    for i = 1: length(kVector)
        cMdl = crossval(model, 'KFold', result(i, 1));
        result(i, 2) = kfoldLoss(cMdl);
    
        if result(i, 2) < minKfoldLoss
            k = result(i, 1);
            minKfoldLoss = result(i, 2);
        end
    end
    
    fig = figure;
    plot(result(:, 1), 100 * result(:, 2));
    title('Procent źle sklasyfikowanych danych przy kross-walidacji')
    ylabel('błędne klasyfikacje [%]');
    xlabel('k - liczba podziałów zbioru treningowego');
end