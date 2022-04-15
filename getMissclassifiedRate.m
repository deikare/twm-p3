function result = getMissclassifiedRate(model, X, Y)
    result = loss(model, X, Y, 'Lossfun', 'classiferror');
end