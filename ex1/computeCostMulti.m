function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% You need to return the following variables correctly
% J = 0.5 / m * sum(((theta(1) + theta(2) * X(:,2) + theta(3) * X(:,3)) - y).^2);
% J = 0.5 / m * sum(((theta(1) + sum(repmat(theta(2:end),[1,m])'.* X(:,2:end), 2)) - y).^2);
J = 0.5 / m * sum((X * theta - y) .^ 2);

% =========================================================================

end
