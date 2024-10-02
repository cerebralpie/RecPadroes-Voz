function [STATS TX_OK X m] = dmc(data, Nr, Ptrain, normtype)
%
% Distance to Mean Classifier (DMC) with optional normalization
%
% INPUTS: * data (matrix): dataset matrix (N x (p+1))
%         * Nr (scalar): Number of runs (Nr>=1)
%         * Ptrain (scalar): Percentage of training data (0 < Ptrain < 100)
%         * normtype (nominal): type of normalization {'none', 'zscore', 'range1', 'range2'}
%           'range1': rescale attributes to [0, 1] range
%           'range2': rescale attributes to [-1, +1] range
%
% OUTPUTS: X (struct) - the data samples separated per class
%          m (struct) - the class centroids
%          STATS (vector) - Statistics of test data (mean, median, min/max, sd)
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

    % Separate into training and testing subsets
    Dtrn = data(1:Ntrn, :);  % Training data
    Dtst = data(Ntrn+1:N, :);  % Testing data

    % Partition of training data into K subsets
    for k = 1:K
        I = find(Dtrn(:, end) == k);  % Find rows with samples from k-th class
        X{k} = Dtrn(I, 1:end-1);  % Data samples from k-th class
        m{k} = mean(X{k})';  % Centroid of the k-th class
    end

    % Testing phase
    correct = 0;  % number of correct classifications
    for i = 1:Ntst
        Xtst = Dtst(i, 1:end-1)';  % Test sample to be classified
        Label_Xtst = Dtst(i, end);  % Actual label of the test sample
        for k = 1:K
            dist(k) = norm(Xtst - m{k});  % Euclidean distance to k-th class
        end
        [dummy, Pred_class] = min(dist);  % Index of the minimum distance class

        if Pred_class == Label_Xtst
            correct = correct + 1;
        end
    end

    TX_OK(r) = 100 * correct / Ntst;  % Recognition rate of r-th run
end

STATS = [mean(TX_OK), min(TX_OK), max(TX_OK), median(TX_OK), std(TX_OK)];

