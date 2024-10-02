function [STATS TX_OK] = knn_1(data, Nr, Ptrain, normtype)
%
% 1-Nearest Neighbor Classifier (1-NN) with optional normalization
%
% INPUTS: * data (matrix): dataset matrix (N x (p+1))
%         * Nr (scalar): Number of runs (Nr>=1)
%         * Ptrain (scalar): Percentage of training data (0 < Ptrain < 100)
%         * normtype (nominal): type of normalization {'none', 'zscore', 'range1', 'range2'}
%           'range1': rescale attributes to [0, 1] range
%           'range2': rescale attributes to [-1, +1] range
%
% OUTPUTS: STATS (vector) - Statistics of test data (mean, median, min/max, sd)
%
[N, p] = size(data);  % Get dataset size (N)

Ntrn = round(Ptrain * N / 100);  % Number of training samples
Ntst = N - Ntrn;  % Number of testing samples

K = max(data(:, end));  % Get the number of classes

switch(normtype)
   case 'none'
      fprintf('No data normalization!\n');
   case 'zscore'
      fprintf('Normalization by Z-score.\n');
      mu = mean(data(:, 1:end-1));
      sigma = std(data(:, 1:end-1));
      data(:, 1:end-1) = (data(:, 1:end-1) - mu) ./ sigma;
   case 'range1'
      fprintf('Normalization to range [0,1].\n');
      mi = min(data(:, 1:end-1));
      ma = max(data(:, 1:end-1));
      data(:, 1:end-1) = (data(:, 1:end-1) - mi) ./ (ma - mi);
   case 'range2'
      fprintf('Normalization to range [-1,+1].\n');
      mi = min(data(:, 1:end-1));
      ma = max(data(:, 1:end-1));
      data(:, 1:end-1) = 2 * ((data(:, 1:end-1) - mi) ./ (ma - mi)) - 1;
end

for r = 1:Nr  % Loop of independent runs
    I = randperm(N);
    data = data(I, :);  % Shuffle rows of the data matrix

    Dtrn = data(1:Ntrn, :);  % Training data
    Dtst = data(Ntrn+1:N, :);  % Testing data

    % Testing phase
    correct = 0;  % number of correct classifications
    for i = 1:Ntst
        Xtst = Dtst(i, 1:end-1);  % Test sample to be classified
        Label_Xtst = Dtst(i, end);  % Actual label of the test sample

        % Find the nearest neighbor in the training set
        for j = 1:Ntrn
            dist(j) = norm(Xtst - Dtrn(j, 1:end-1));  % Euclidean distance to training samples
        end
        [dummy, idx] = min(dist);  % Index of the nearest neighbor

        Pred_class = Dtrn(idx, end);  % Class of the nearest neighbor

        if Pred_class == Label_Xtst
            correct = correct + 1;
        end
    end

    TX_OK(r) = 100 * correct / Ntst;  % Recognition rate of r-th run
end

STATS = [mean(TX_OK), min(TX_OK), max(TX_OK), median(TX_OK), std(TX_OK)];

