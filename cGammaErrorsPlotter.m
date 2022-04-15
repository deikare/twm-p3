function figTrain = cGammaErrorsPlotter(costs, gammas, errors)
    figTrain = figure;
    [x, y] = meshgrid(costs, gammas);
    surf(x, y, 100 * errors);
    set(gca, 'XScale', 'log')
    set(gca, 'YScale', 'log');
    xlabel('c');
    ylabel('\gamma');
    zlabel('błędne klasyfikacje [%]');
    title('Procent błędu kross-walidacji w zależności od c oraz \gamma')
    view(52, 24);
end