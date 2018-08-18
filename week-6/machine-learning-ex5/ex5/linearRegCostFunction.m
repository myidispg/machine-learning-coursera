function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% X = [12x2] theta = [2*1]
X_theta = X * theta; % 12x2 * 2*1 = 12x1;
err = sum((X_theta - y) .^ 2); % (12x1) ^ 2 = 12x1 => sum(12x1) = 1

regularized = lambda * sum(theta(2:end) .^ 2);

J = (err + regularized)/(2*m);

grad = lambda * theta / m; % 1 * [2x1] / 1 = [2x1]
grad(1) = 0;
grad = grad + sum((X_theta - y) .* X)' / m; % 2x1 + sum(12x1 .* 12x1) = [2x1];

% =========================================================================

grad = grad(:);

end
