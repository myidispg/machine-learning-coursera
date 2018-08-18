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
X_bias = [ones(size(X,1), 1) X]; % Added bias to all 5000 examples(dimension = 5000x401)
Z2 = X_bias * Theta1'; % Computed activations of next layer(dimension = 5000x25)
a2 = sigmoid(Z2); % brought all activation b/w 0 and 1(dimension = 5000x25)
a2_bias = [ones(size(a2, 1), 1) a2]; % Added bias to a2 layer(dimension = 5000x26)
Z3 = a2_bias * Theta2'; % Computed activations of final layer(dimension = 5000x10)
H = sigmoid(Z3); % brought all outputs b/w 0 and 1(dimension = 5000x10)
[val index] =max(H, [], 2);
for k = 1: num_labels
    yy = y == k; % this puts all those values of y where values are equal to k in yy
    % yy has 1 in all those indices where y matrix has a value equal to k.
    hh = H(:, k); % hh = finding the predicted output for all those indices in H where y has value equal to k
    % if k is 3, hh has probability of 3 being the answer in all 5000 training examples.
    % both hh and yy have dimension = 5000x1.
    err = - yy .* log(hh) - (1-yy) .* log(1-hh);
    J = J + sum(err); 
    % if k has value of 5, J has now added the cost of all those predictions, where value should have been 5.
end
J = J/m;
 
% Regularizing the cost function excluding j = 0
J = J + lambda / (2*m) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
% sum(sum(Theta1(:, 2:end) .^ 2)) explanation - .^ 2 squares all the values at their indices.
% sum(Theta1(:, 2:end) .^ 2) returns a vector where each column is the sum of values in all the rows of that column
% then summing it again returns a single value of the sum of all the columns.

% --------Back Propogation---------------------------------------------
for t = 1:m
    a_1 = X(t, :); % [1 * 400]
    z_2 = [1, a_1] * Theta1'; % 1x401 * 401x25 = [1x25]
    a_2 = sigmoid(z_2); % [1x25]
    z_3 = [1, a_2] * Theta2'; % 1x26 * 26x10 = [1x10]
    a_3 = sigmoid(z_3); % [1x10]

    delta_3 = a_3; 
    delta_3(y(t)) = delta_3(y(t)) - 1; % [1x10]
    % if y has value 5 for t-th training example, delta_3(5) will be 0 otherwise all else will be treated as cost. 
    delta_2 = delta_3 * Theta2; % 1x10 * 10x26 = [1*26]
    % removing bias unit from delta_2
    delta_2 = delta_2(2:end); % [1x25]
    delta_2 = delta_2 .* sigmoidGradient(z_2);

    Theta2_grad = Theta2_grad + delta_3' * [1 a_2]; % 10x26 + (10x1 * 10x26)
    Theta1_grad = Theta1_grad + delta_2' * [1 a_1]; % 25x401 + (25x1 * 1x401)
end
 Theta1_grad = Theta1_grad/m; % [25x401]
 Theta2_grad = Theta2_grad/m; % [10x26]
    %-------Regularization-------------
 theta1_temp = Theta1; % [25x401]
 theta2_temp = Theta2; % [10x26]
 theta1_temp(:, 1) = 0; % [25x401]
 theta2_temp(:, 1) = 0; % [10x26]
 % Final Theta gradients.
 Theta1_grad = Theta1_grad + (lambda/m) * theta1_temp; % [25x401]
 Theta2_grad = Theta2_grad + (lambda/m) * theta2_temp; % [10x26]

% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
