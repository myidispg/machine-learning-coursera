function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));
% Y = 1682x943 1682 movies 943 users
% X = 1682x100, Theta = 943x100
% R = 1682x943 matrix
% X_grad = 1682x943, Theta_grad = 943x100
%
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
% for only those where R = 1, the sum will be added.
J = sum(sum(R .* (X*Theta' - Y) .^ 2)) * (1/2) + (lambda/2) * (sum(sum(Theta.^ 2)) + sum(sum(X .^ 2)));

for i=1:num_movies
    idx = find(R(i, :) == 1); % 1x452
    tempTheta = Theta(idx, :);  
    tempY = Y(i, idx);
    X_grad(i, :) = (X(i, :) * tempTheta' - tempY) * tempTheta + lambda * X(i, :);
end
% tempTheta = R .* Theta;
% X_grad = sum(sum(sum(R .* (X*Theta' - Y) .^ 2)) .* Theta);
for j = 1: num_users
    idy = find(R(:, j) == 1);  
    Theta_temp = Theta(j, :); % 1 * num_features
    Y_tem = Y(idy, j); % idy * 1
    X_tem = X(idy, :); % idy * num_features
    Theta_grad(j,:) = (X_tem * Theta_temp' - Y_tem)' * X_tem + lambda * Theta(j, :);
end
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end