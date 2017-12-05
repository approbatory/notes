function prob = softmax_func(X)
%% softmax function
%% input parameters: x = m x c matrix, m is the block size, 
% n is the number of classes
%% output parameters: : m x c matrix
%% Your code here
prob = exp(X - max(X,2));
prob = prob ./ sum(prob,2);
end
