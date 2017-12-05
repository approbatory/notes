function loss = cross_entropy(pred, y_onehot)
%% function for computing cross entropy loss
%% input parameters: 
% pred and y_onehot are both m x c matrix, the element at position (i j)
% corresponds to the probability and actual label of sample i for class j.
%% output: average cross entropy loss
%% Your code here
%[~, y_index] = max(y_onehot,[],2);
loss = -mean(log(pred(y_onehot==1)));
%loss = -mean(log(pred(:,y_index)));
%loss = -mean(sum(y_onehot .* log(pred),2));
%loss = -1/size(pred,1)*trace(y_onehot * log(pred)');
end