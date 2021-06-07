function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%

% method 1
X_poly = [];
for i = 1:p
    X_poly = [X_poly X.^i];
end

% method 2
% https://github.com/emersonmoretto/mlclass-ex5/blob/master/polyFeatures.m
% X_poly = zeros(numel(X), p);
% for i=1:p,
% 	X_poly(:,i) = X(:,1).^i;
% end

% =========================================================================

end
