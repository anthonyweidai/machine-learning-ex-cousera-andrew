function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    % method 1
    % temp = theta; %% remember simultaneously update theta
    % for i = 1:length(temp)
    %     theta(i) = temp(i) - alpha / m * sum((X * temp - y) .* X(:,i));
    % end
    
    % method 2
    delta = sum((X * theta - y) .* X)';
    theta = theta - alpha / m * delta;
    
    % method 3
    %%%%%%%%% https://gist.github.com/retnuh/1312646
    %delta = ((theta' * X' - y') * X)';
    %theta = theta - alpha / m * delta;

    % ============================================================
    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
