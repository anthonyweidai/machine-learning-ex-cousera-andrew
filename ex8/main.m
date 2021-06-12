clear, close all
% %%
% % The following command loads the dataset. You should now have the variables X, Xval, yval in your environment
% load('ex8data1.mat');
% 
% % Visualize the example dataset
% plot(X(:, 1), X(:, 2), 'bx');
% axis([0 30 0 30]);
% xlabel('Latency (ms)');
% ylabel('Throughput (mb/s)');
% 
% %  Estimate mu and sigma2
% [mu, sigma2] = estimateGaussian(X);
% 
% %  Returns the density of the multivariate normal at each data point (row) of X
% p = multivariateGaussian(X, mu, sigma2);
% 
% %  Visualize the fit
% visualizeFit(X,  mu, sigma2);
% xlabel('Latency (ms)');
% ylabel('Throughput (mb/s)');

% %%
% pval = multivariateGaussian(Xval, mu, sigma2);
% 
% [epsilon, F1] = selectThreshold(yval, pval);
% fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
% fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
% 
% %  Find the outliers in the training set and plot the
% outliers = find(p < epsilon);
% 
% %  Visualize the fit
% visualizeFit(X,  mu, sigma2);
% xlabel('Latency (ms)');
% ylabel('Throughput (mb/s)');
% %  Draw a red circle around those outliers
% hold on
% plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
% hold off

% %  Loads the second dataset. You should now have the variables X, Xval, yval in your environment
% load('ex8data2.mat');
% 
% %  Apply the same steps to the larger dataset
% [mu, sigma2] = estimateGaussian(X);
% 
% %  Training set 
% p = multivariateGaussian(X, mu, sigma2);
% 
% %  Cross-validation set
% pval = multivariateGaussian(Xval, mu, sigma2);
% 
% %  Find the best threshold
% [epsilon, F1] = selectThreshold(yval, pval);
% fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
% fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
% fprintf('# Outliers found: %d\n', sum(p < epsilon));

% %%
% % Load data
% load('ex8_movies.mat');
% 
% % From the matrix, we can compute statistics like average rating.
% fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', mean(Y(1, R(1, :))));
% %  We can "visualize" the ratings matrix by plotting it with imagesc
% imagesc(Y);
% ylabel('Movies');
% xlabel('Users');
% 
% %  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
% load('ex8_movieParams.mat');
% 
% %  Reduce the data set size so that this runs faster
% num_users = 4; num_movies = 5; num_features = 3;
% X = X(1:num_movies, 1:num_features);
% Theta = Theta(1:num_users, 1:num_features);
% Y = Y(1:num_movies, 1:num_users);
% R = R(1:num_movies, 1:num_users);
% 
% %  Evaluate cost function
% J = cofiCostFunc([X(:); Theta(:)],  Y, R, num_users, num_movies,num_features, 0);
% fprintf('Cost at loaded parameters: %f ',J);
% 
% %  Check gradients by running checkNNGradients
% checkCostFunction;
% 
% %  Evaluate cost function
% J = cofiCostFunc([X(:); Theta(:)], Y, R, num_users, num_movies, num_features, 1.5);      
% fprintf('Cost at loaded parameters (lambda = 1.5): %f',J);
% 
% %  Check gradients by running checkNNGradients
% checkCostFunction(1.5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Optional %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% Load movvie list
movieList = loadMovieList();

% Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings(1) = 4;
% Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings(98) = 2; 

% We have selected a few movies we liked / did not like and the ratings we gave are as follows:
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;

fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
    end
end

%%
%  Load data
load('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 943 users
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj','on','MaxIter',100);

% Set Regularization
lambda = 10;
theta = fmincg(@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, num_features,lambda)), initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), num_users, num_features);

%%
p = X * Theta';
my_predictions = p(:,1) + Ymean;

movieList = loadMovieList();

[r, ix] = sort(my_predictions,'descend');
for i=1:10
    j = ix(i);
    if i == 1
        fprintf('\nTop recommendations for you:\n');
    end
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), movieList{j});
end
for i = 1:length(my_ratings)
    if i == 1
        fprintf('\n\nOriginal ratings provided:\n');
    end
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
    end
end