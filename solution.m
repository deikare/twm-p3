%% Parametry działania
% Powtarzalne wyniki
close all ;
rng('default') ;

% Liczba obrazów treningowych na klasę
cnt_train = 70 ;

% Liczba obrazów testowych na klasę
cnt_test = 15;

% Liczba obrazów walidacyjnych na klasę
cnt_val = 15;

% Wybrane klasy obiektów
img_classes = {'deli', 'greenhouse', 'bathroom'};

% Liczba cech wybierana na każdym obrazie
feats_det = 100;

% Metoda wyboru cech (true - jednorodnie w całym obrazie, false - najsilniejsze)
feats_uniform = true;

% Wielkość słownika
words_cnt = 30 ;

% Detekcja cech
% Ładowanie pełnego zbioru danych z automatycznym podziałem na klasy
% Zbiór danych pochodzi z publikacji: A. Quattoni, and A.Torralba. <http://people.csail.mit.edu/torralba/publications/indoor.pdf 
% _Recognizing Indoor Scenes_>. IEEE Conference on Computer Vision and Pattern 
% Recognition (CVPR), 2009.
% 
% Pełny zbiór dostępny jest na stronie autorów: <http://web.mit.edu/torralba/www/indoor.html 
% http://web.mit.edu/torralba/www/indoor.html>

imds_full = imageDatastore("Images/", "IncludeSubfolders", true, "LabelSource", "foldernames");
%countEachLabel(imds_full)

% Wybór przykładowych klas i podział na zbiór treningowy i testowy
[imds, imtest, imval] = splitEachLabel(imds_full, cnt_train, cnt_test, cnt_val, 'Include', img_classes);

% Wyznaczenie punktów charakterystycznych we wszystkich obrazach zbioru treningowego
files_cnt = length(imds.Files);
all_points = cell(files_cnt, 1);
total_features = 0;

for i=1:files_cnt
    I = readImage(imds.Files{i});
    all_points{i} = getFeaturePoints(I, feats_det, feats_uniform);
    total_features = total_features + length(all_points{i});
end

% Przygotowanie listy przechowującej indeksy plików i punktów charakterystycznych
file_ids = zeros(total_features, 2);
curr_idx = 1;
for i=1:files_cnt
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 1) = i;
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 2) = 1:length(all_points{i});
    curr_idx = curr_idx + length(all_points{i});
end

% Obliczenie deskryptorów punktów charakterystycznych
all_features = zeros(total_features, 64, 'single');
curr_idx = 1;
for i=1:files_cnt
    I = readImage(imds.Files{i});
    curr_features = extractFeatures(im2gray(I), all_points{i});
    all_features(curr_idx:curr_idx+length(all_points{i})-1, :) = curr_features;
    curr_idx = curr_idx + length(all_points{i});
end

% Tworzenie słownika

% Klasteryzacja punktów 
[idx, words, sumd, D] = kmeans(all_features, words_cnt, "MaxIter", 10000);
% Wizualizacja wyliczonych słów

% Wyznaczenie histogramów słów dla każdego obrazu treningowego
file_hist = zeros(files_cnt, words_cnt);
for i=1:files_cnt
    file_hist(i,:) = histcounts(idx(file_ids(:,1) == i), (1:words_cnt+1)-0.5, 'Normalization', 'probability');
end
% Wyznaczenie histogramów słów dla każdego obrazu walidacyjnego
val_hist = zeros(length(imval.Files), words_cnt);
for i=1:length(imval.Files)
    I = readImage(imval.Files{i});
    pts = getFeaturePoints(I, feats_det, feats_uniform);
    feats = extractFeatures(rgb2gray(I), pts);
    val_hist(i,:) = wordHist(feats, words);
end

% Wyznaczenie histogramów słów dla każdego obrazu testowego
test_hist = zeros(length(imtest.Files), words_cnt);
for i=1:length(imtest.Files)
    I = readImage(imtest.Files{i});
    pts = getFeaturePoints(I, feats_det, feats_uniform);
    feats = extractFeatures(rgb2gray(I), pts);
    test_hist(i,:) = wordHist(feats, words);
end

%% wieloklasowy klasyfikator SVM z jądrem Gaussa
t = templateSVM("KernelFunction", "gaussian");
Mdl = fitcecoc(file_hist, imds.Labels, 'Learners', t);
[label,NegLoss,PBScore] = predict(Mdl, test_hist);
numberOfMissmatches(imtest.Labels, label)
percentageOfMissmatches(imtest.Labels, label)

cMdl = crossval(Mdl);

[label,NegLoss,PBScore] = predict(cMdl, test_hist);
numberOfMissmatches(imtest.Labels, label)
percentageOfMissmatches(imtest.Labels, label)

%% Funkcje pomocnicze

function result = percentageOfMissmatches(labels, prediction)
    result = numberOfMissmatches(labels, prediction) * 100 / length(labels);
end

function result = numberOfMissmatches(labels, prediction)
    result = 0;
    for i = 1 : size(labels)
        if labels(i) ~= prediction(i)
            result = result + 1;
        end
    end
end

function P = getPatch(I, pt, scale, scale_factor)
    x1 = round(pt(1) - 0.5*scale*scale_factor);
    x2 = round(pt(1) + 0.5*scale*scale_factor);
    y1 = round(pt(2) - 0.5*scale*scale_factor);
    y2 = round(pt(2) + 0.5*scale*scale_factor);
    
    [x1, x2, y1, y2] = clipInside(x1, x2, y1, y2, size(I, 1), size(I, 2));
    
    P = imresize(I(y1:y2, x1:x2, :), [64 64]);
end

function [xr1, xr2, yr1, yr2] = clipInside(x1, x2, y1, y2, rows, cols)
    xr1 = min(max(x1, 1), cols);
    xr2 = min(max(x2, 1), cols);
    yr1 = min(max(y1, 1), rows);
    yr2 = min(max(y2, 1), rows);
end

function pts = getFeaturePoints(I, pts_det, pts_uniform)
    if size(I, 3) > 1
        I2 = rgb2gray(I);
    else
        I2 = I;
    end
    
    pts = detectSURFFeatures(I2, 'MetricThreshold', 100);
    if pts_uniform
        pts = selectUniform(pts, pts_det, size(I));
    else
        pts = pts.selectStrongest(pts_det);
    end
end

function h = wordHist(feats, words)
    words_cnt = size(words, 1);
    dis = pdist2(feats, words, 'squaredeuclidean');
    [~, lbl] = min(dis, [], 2);
    h = histcounts(lbl, (1:words_cnt+1)-0.5, 'Normalization', 'probability');
end

function [h, P] = visSingleImage(I, pts, feats, words)
    words_cnt = size(words, 1);
    dis = pdist2(feats, words, 'squaredeuclidean');
    [dis, lbl] = min(dis, [], 2);
    [~, ids] = sort(dis);
    h = histcounts(lbl, (1:words_cnt+1)-0.5, 'Normalization', 'probability');
    P = zeros(words_cnt*64, 30*64, 3, 'uint8');
    pos = zeros(words_cnt, 1);
    for i=1:size(feats, 1)
        id = ids(i);
        x = pos(lbl(id)) * 64;
        pos(lbl(id)) = min(pos(lbl(id)) + 1, 29);
        y = (lbl(id)-1) * 64;
        pat = getPatch(I, pts.Location(id, :), pts.Scale(id), 12);
        pat = insertText(pat, [2, 2], dis(id), 'FontSize', 10, 'BoxOpacity', 0);
        pat = insertText(pat, [1, 1], dis(id), 'FontSize', 10, 'BoxOpacity', 0, 'TextColor', 'white');
        P(y+1:y+64, x+1:x+64, :) = pat;
    end
end

% Wczytanie obrazu i przeskalowanie jeśli jest zbyt duży
function I = readImage(path)
    I = imread(path);
    if size(I,2) > 640
        I = imresize(I, [NaN 640]);
    end
end