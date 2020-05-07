function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Adding ones to the input data
X = [ones(size(X,1),1) X];

% Computing the hiddern layer matrix (Should be 5000 rows X 25 cols)
a2 = sigmoid(X * Theta1');

% Adding ones to a2 i.e. the hidden layer (Should be 5000 rows X 26 cols)
a2 = [ones(size(a2,1),1) a2];

% Computing the output layer (Should be 5000 rows X 10 cols)
a3 = sigmoid(a2 * Theta2');

% Getting the indices of the max values
[val, index] = max(a3');

% Transposing the values to return the prediction
p = index';


% =========================================================================


end
