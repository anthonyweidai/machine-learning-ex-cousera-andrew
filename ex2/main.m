clear, clc, close all
%% Load Data
% The first two columns contain the exam scores and the third column contains the label.
data = load('ex2data1.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

%%%%%% Plot the data with + indicating (y = 1) examples and o indicating (y = 0) examples.
plotData(X, y);
 
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')

%%%%%% test your sigmoid function
sigmoid(0)

%% Initialize the data
% Setup the data matrix appropriately
[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Initialize the fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display the initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);
fprintf('Cost at initial theta (zeros): %f\n', cost);
disp('Gradient at initial theta (zeros):'); 
disp(grad);

% Compute and display cost and gradient with non-zero theta
test_theta = [-24; 0.2; 0.2];
[cost, grad] = costFunction(test_theta, X, y);
fprintf('\nCost at non-zero test theta: %f\n', cost);
disp('Gradient at non-zero theta:'); disp(grad);

%%  Set options for fminunc
options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta
fprintf('Cost at theta found by fminunc: %f\n', cost);
disp('theta:');disp(theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
% Add some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

%%  Predict probability for a student with score 45 on exam 1  and score 85 on exam 2 
prob = sigmoid([1 45 85] * theta);
fprintf('For a student with scores 45 and 85, we predict an admission probability of %f\n\n', prob);
% Compute accuracy on our training set
p = predict(theta, X);
% fprintf('Train Accuracy: %f\n', length(find(abs(p - y) <= exp(-4))) / length(y));
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

%%%%%% Regularized logistic regression
% %%  The first two columns contains the X values and the third column
% %  contains the label (y).
% data = load('ex2data2.txt');
% X = data(:, [1, 2]); y = data(:, 3);
% 
% plotData(X, y);
% % Put some labels 
% hold on;
% % Labels and Legend
% xlabel('Microchip Test 1')
% ylabel('Microchip Test 2')
% % Specified in plot order
% legend('y = 1', 'y = 0')
% hold off;
% 
% %% Add Polynomial Features
% % Note that mapFeature also adds a column of ones for us, so the intercept term is handled
% X = mapFeature(X(:,1), X(:,2));
% 
% % Initialize fitting parameters
% initial_theta = zeros(size(X, 2), 1);
% 
% % Set regularization parameter lambda to 1
% lambda = 1;
% 
% %% Compute and display initial cost and gradient for regularized logistic regression
% [cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
% fprintf('Cost at initial theta (zeros): %f\n', cost);
% fprintf('Expected cost (approx): 0.693\n');
% fprintf('Gradient at initial theta (zeros) - first five values only:\n');
% fprintf(' %f \n', grad(1:5));
% fprintf('Expected gradients (approx) - first five values only:\n');
% fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');
% 
% % Compute and display cost and gradient with all-ones theta and lambda = 10
% test_theta = ones(size(X,2),1);
% [cost, grad] = costFunctionReg(test_theta, X, y, 10);
% fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
% fprintf('Expected cost (approx): 3.16\n');
% fprintf('Gradient at test theta - first five values only:\n');
% fprintf(' %f \n', grad(1:5));
% fprintf('Expected gradients (approx) - first five values only:\n');
% fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');
% 
% %%
% % Initialize fitting parameters
% initial_theta = zeros(size(X, 2), 1);
% 
% lambda = 1;
% % Set Options
% options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 1000);
% 
% % Optimize
% [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
% 
% %%
% plotDecisionBoundary(theta, X, y)
% 
% % Compute accuracy on our training set
% p = predict(theta, X);
% 
% fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);