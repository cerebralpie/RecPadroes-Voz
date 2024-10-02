% Max Correlation Classifier for Voice Detection
% Last modification: 20/01/2024

function [STATS TX_OK W] = maxCorr(data, Nr, Ptrain, normtype)

  clear; clc; close all;

  % Load preprocessed data
  [N, p] = size(data);              % N = number of samples, p = number of features + label

  % Separate features (X) and labels (Y)
  X = data(:, 1:end-1);             % Features (all columns except the last)
  Y = data(:, end);                 % Labels (last column)

  % Parameters
  K = max(Y);                       % Number of classes

  % Pre-allocate variables for statistics
  STATS = zeros(Nr, 5);             % Statistics over multiple runs

  switch(normtype)
   case 'none'
      fprintf('No data normalization!\n' );
   case 'zscore'
      fprintf('Normalization by Z-score.\n' );
      data(:,1:end-1) = (data(:,1:end-1) - med)./dp;
   case 'range1'
      fprintf('Normalization to range [0,1].\n' );
      data(:,1:end-1) = (data(:,1:end-1) - mi)./(ma-mi);
   case 'range2'
      fprintf('Normalization to range [-1,+1].\n' );
      data(:,1:end-1) = 2*((data(:,1:end-1) - mi)./(ma-mi))-1;
   end
  
  for r = 1:Nr
      % Shuffle the data
      idx = randperm(N); 
      data_shuffled = data(idx, :);

      % Split data into training and testing sets
      Ntrn = round(Ptrain * N / 100);   % Number of training samples
      Xtrn = data_shuffled(1:Ntrn, 1:end-1); % Training features
      Ytrn = data_shuffled(1:Ntrn, end);    % Training labels
      Xtst = data_shuffled(Ntrn+1:end, 1:end-1); % Testing features
      Ytst = data_shuffled(Ntrn+1:end, end);    % Testing labels
      Ntst = size(Xtst, 1);              % Number of test samples

      % Compute centroids (means) for each class
      centroids = zeros(K, size(Xtrn, 2));
      for k = 1:K
          centroids(k, :) = mean(Xtrn(Ytrn == k, :), 1);  % Mean vector for class k
      end

      % Testing phase
      correct = 0;  % Counter for correct classifications
      for i = 1:Ntst
          % Get test sample
          test_sample = Xtst(i, :);

          % Compute correlation with each class centroid
          corr_values = zeros(1, K);
          for k = 1:K
              corr_values(k) = corr(test_sample', centroids(k, :)');  % Correlation with class k centroid
          end

          % Find class with maximum correlation
          [~, predicted_class] = max(corr_values);

          % Check if the prediction is correct
          if predicted_class == Ytst(i)
              correct = correct + 1;
          end
      end

      % Compute recognition rate for this run
      TX_OK(r) = 100 * correct / Ntst;
      
  end
  STATS = [mean(TX_OK), min(TX_OK), max(TX_OK), median(TX_OK), std(TX_OK)];
end