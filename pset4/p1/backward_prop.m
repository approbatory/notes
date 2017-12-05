function [dW1, db1, dW2, db2] = backward_prop(X, y_onehot, W1, b1, W2, b2, lambda)
%% backward propagation for our 1 layer network
%% input parameters
% X is our m x n dataset, where m = number of samples, n = number of
% features
% y is the length m label vector for each sample
% W1 is our h1 x n weight matrix, where h1 = number of hidden units in
% layer 1
% b1 is the length h1 column vector of bias terms associated with layer 1
% W2 is the c x h1 weight matrix, where c = number of classes
% b2 is the length h2 column vector of bias terms associated with the output
%% output parameters
% returns the gradient of W1, b1, W2, b2 as dW1, db1, dW2, db2
%% Your code here
[h_output, prob, loss] = forward_prop(X, y_onehot, W1, b1, W2, b2, lambda);
%%mean_h = mean(h_output);
%%delta_2 = mean(prob - y_onehot)';
%%db2 = delta_2;
%%dW2 = delta_2*mean_h + 2*lambda*W2;
%%delta_1 = W2'*delta_2 .* mean_h' .* (1-mean_h');
%%db1 = delta_1;
%%dW1 = delta_1*mean(X) + 2*lambda*W1;
B = size(X,1);
delta_2 = (prob - y_onehot)';
db2 = sum(delta_2,2) / B;
dW2 = delta_2*h_output / B + 2*lambda*W2;
delta_1 = W2'*delta_2 .* h_output' .* (1-h_output');
db1 = sum(delta_1,2) / B;
dW1 = delta_1*X / B + 2*lambda*W1;
end