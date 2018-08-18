function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------Cost Function---------------------
X_bias = [ones(size(X, 1), 1) X]; % 5000x401
z2 = X_bias * Theta1'; % 5000x401 * 401x25 = 5000x25
a2 = sigmoid(z2); % 5000x25
a2_bias = [ones(size(a2, 1), 1) a2]; % 5000x26
z3 = a2_bias * Theta2'; % 5000x26 * 26x10 = 5000x10
H = sigmoid(z3);
for k = 1:num_labels
    yy = y == k; % 5000x1
    hh = H(:, k); % 5000x1
    err = (-yy .* log(hh)) - ((1-yy) .* log(1-hh));
    J = J + sum(err);
end
J = J/m;

J = J + (lambda/(2*m)) * ((sum(sum(Theta1(:, 2:end) .^ 2))) + (sum(sum(Theta2(:, 2:end) .^ 2))));
% --------Back Propogation---------------------------------------------
for t=1:m
    a1 = X(t, :); % 1x400
    a1_bias = [1 a1]; % 1x401
    z2 = a1_bias * Theta1'; % 1x401 * 401x25 = 1x25
    a2 = sigmoid(z2); % 1x25
    a2_bias = [1 a2]; % 1x26
    z3 = a2_bias * Theta2'; % 1x26 * 26x10 = 1x10
    H = sigmoid(z3); % 1x10

    delta3 = H; % 1x10
    delta3(y(t)) = delta3(y(t)) - 1; % 1x10

    delta2 = delta3 * Theta2; % 1x10 * 10x26 = 1x26
    delta2 = delta2(2:end); % 1x25
    delta2 = delta2 .* sigmoidGradient(z2); % 1x25

    Theta1_grad = Theta1_grad + delta2' * a1_bias; % 25x401 +  25x1 * 1x401 = 25x401
    Theta2_grad = Theta2_grad + delta3' * a2_bias; % 10x26 + 10x1 * 1x26 = 10x26
end
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;
    %-------Regularization-------------
theta1_temp = Theta1_grad; % 25x401
theta2_temp = Theta2_grad; % 10x26
theta1_temp(:, 1) = 0; % 25x401 
theta2_temp(:, 1) = 0; % 10x26

% Final Theta gradients.
Theta1_grad = Theta1_grad + (lambda/m) * theta1_temp; % 25x401 + 25x401 = 25x401;
Theta2_grad = Theta2_grad + (lambda/m) * theta2_temp; % 10X26 + 10X26 = 10X26;
% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
