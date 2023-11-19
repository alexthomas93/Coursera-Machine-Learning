function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
[p, q] = meshgrid(params, params);
param_combs = [p(:) q(:)];
m = length(param_combs(:, 1));
errs = zeros(m, 1);
for i = 1:m
  mdl = svmTrain(X, y, param_combs(i, 1), ... 
        @(x1, x2) gaussianKernel(x1, x2, param_combs(i, 2)));
  preds = svmPredict(mdl, Xval);
  errs(i) = mean(double(preds ~= yval));
endfor
[mn, imn] = min(errs);
C = param_combs(imn, 1);
sigma = param_combs(imn, 2);
% =========================================================================

end
