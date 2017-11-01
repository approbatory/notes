function error = nb_test_func(log_prob_spam, log_prob_nospam,...
    log_spam_word_probs, log_nospam_word_probs)
[spmatrix, tokenlist, category] = readMatrix('MATRIX.TEST');

testMatrix = full(spmatrix);
numTestDocs = size(testMatrix, 1);
numTokens = size(testMatrix, 2);

% Assume nb_train.m has just been executed, and all the parameters computed/needed
% by your classifier are in memory through that execution. You can also assume 
% that the columns in the test set are arranged in exactly the same way as for the
% training set (i.e., the j-th column represents the same token in the test data 
% matrix as in the original training data matrix).

% Write code below to classify each document in the test set (ie, each row
% in the current document word matrix) as 1 for SPAM and 0 for NON-SPAM.

% Construct the (numTestDocs x 1) vector 'output' such that the i-th entry 
% of this vector is the predicted class (1/0) for the i-th  email (i-th row 
% in testMatrix) in the test set.
output = zeros(numTestDocs, 1);

%---------------
% YOUR CODE HERE
for i = 1:numTestDocs
    x = testMatrix(i,:);
    log_posterior_spam = log_prob_spam + sum(log_spam_word_probs .* x);
    log_posterior_nospam = log_prob_nospam + sum(log_nospam_word_probs .* x);
    if log_posterior_spam > log_posterior_nospam
        output(i) = 1;
    else
        output(i) = 0;
    end
end

%---------------


% Compute the error on the test set
y = full(category);
y = y(:);
error = sum(y ~= output) / numTestDocs;

%Print out the classification error on the test set
%fprintf(1, 'Test error: %1.4f\n', error);

end

