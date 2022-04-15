% Oblicza funkcję hipotezy regresji logistyczną (wielomianowej) przy użyciu
% macierzy wag W, stosując funkcję softmax
function evals = hfun1(W,features)
    temp = exp(features * W) ;
    evals = temp ./ sum(temp,2) ;
end