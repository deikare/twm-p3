function [W, cost] = trainsimple(inputfeatures, inputlabels, regcoeff)

featurestrain = inputfeatures ;
labelstrain = inputlabels == unique(inputlabels)' ;

% Minimalizacja funkcji
W = rand(size(featurestrain,2), size(labelstrain,2)) ;
options = optimoptions('fminunc','Display','none', 'MaxFunctionEvaluations', 10e9) ;
[w,cost,exitflag] = fminunc(@(w) costfun1(w,featurestrain,labelstrain,regcoeff),W(:),options) ;
W = reshape(w,size(featurestrain,2), size(labelstrain,2)) ;

end



