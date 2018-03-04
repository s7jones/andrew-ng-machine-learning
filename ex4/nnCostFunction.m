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



for k = 1:num_labels
    
    ylog = (y == k);
    
    J = J + (1 / m) * (-ylog' * log(h(X,k,Theta1,Theta2)) - (1 - ylog)' * log(1 - h(X,k,Theta1,Theta2)));

end

if (isOctave)
  f = @sumsqr_octave;
else
  f = @sumsqr;
end

J = J + (lambda / (2 * m)) * (f(Theta1(:,2:end)) + f(Theta2(:,2:end)));

for t = 1:m
   
    % 1.
    a_1 = X(t,:);
    
    a_1 = [1, a_1];
    
    z_2 = a_1 * Theta1';
    
    a_2 = g(z_2);

    a_2 = [1, a_2];

    a_3 = g(a_2 * Theta2');
    
    for k = 1:num_labels
        d_3(k) = a_3(k) - (y(t)==k);
    end
    
    Theta2_d_3 = (Theta2' * d_3')'; % remove d_0^(2)

    d_2 = Theta2_d_3(2:end) .* (sigmoidGradient(z_2));
    
    Theta1_grad = Theta1_grad + d_2' * a_1;
    Theta2_grad = Theta2_grad + d_3' * a_2;       
end

Theta1_nobias = [zeros(size(Theta1, 1),1),  Theta1(:,2:end)];
Theta2_nobias = [zeros(size(Theta2, 1),1),  Theta2(:,2:end)];

Theta1_grad = Theta1_grad + lambda * Theta1_nobias;
Theta2_grad = Theta2_grad + lambda * Theta2_nobias;

Theta1_grad = Theta1_grad * (1/m);
Theta2_grad = Theta2_grad * (1/m);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function g = g(z)
    g = sigmoid(z);
end

function h = h(X, k, Theta1, Theta2)
    m = size(X,1);

    X = [ones(m, 1) X];

    %a2 = zeros(size(X,1), size(Theta1,1));

    a2 = g(X * Theta1');

    a2 = [ones(m, 1) a2];

    %a3 = zeros(size(X,1), size(Theta2,1));

    a3 = g(a2 * Theta2');
    
    h = a3(:,k);
end

% function z2 = z2(X, Theta1)
%     m = size(X,1);
% 
%     X = [ones(m, 1) X];
% 
%     z2 = X * Theta1';
% end