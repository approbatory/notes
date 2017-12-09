function [h_output, prob, loss] = forward_prop(X, y_onehot, W1, b1, W2, b2, lambda)
%% forward propagation for our 1 layer network
%% input parameters
% X is our m x n dataset, where m = number of samples, n = number of
% features
% W1 is our h1 x n weight matrix, where h1 = number of hidden units in
% layer 1
% b1 is the length h1 column vector of bias terms associated with layer 1
% W2 is the c x h1 weight matrix, where c = number of classes
% b2 is the length h2 column vector of bias terms associated with the output
%% output parameters
% returns a probability matrix of dimension m x c, where the element in
% position (i, j) corresponds to the probability that sample i is in class
% j
%% Your code here
%%%h_output = sigmoid_func(X * W1' + b1');
%%%z = h_output * W2' + b2';
%%%prob = softmax_func(z);
%%%loss = cross_entropy(prob, y_onehot); %+ lambda*(trace(W1*W1') + trace(W2*W2'));
%%%if any(any(isinf(loss)))
%%%    %warning('found inf');
%%%end
M = size(X,1);
X = X';
y_onehot = y_onehot';
h_output = sigmoid_func(b1 + W1*X);
z = b2 + W2*h_output;
prob = exp(z - max(z));
prob = prob./sum(prob);
loss = -y_onehot(:)'*log(prob(:)) / M;
prob = prob';
end