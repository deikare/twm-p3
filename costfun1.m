

% Oblicza funkcję kosztu przy użyciu funkcji entropii krzyżowej
function cost = costfun1(w,features,labels,lambda)
    W = reshape(w,size(features,2),size(labels,2)) ;
    evals = hfun1(W, features) ;
    logloss = -labels .* log(evals) ;
    logloss(labels == 0) = 0 ; % unikamy NaN dla lograrytmów z 0
    cost = sum(sum(logloss))/size(features,1) ;
    cost = cost + lambda * sum(w(1:end-1).^2)/size(features,1) ;
    %ws = W(1:end-1,:) ;
    %ws = ws(:) ;
    %cost = cost + 0.000001*(var(ws) - 1)^2;
end