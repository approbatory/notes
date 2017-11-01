function [log_prob_spam, log_prob_nospam,...
    log_spam_word_probs, log_nospam_word_probs]...
    = nb_train_func(train_fname)


%[spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN');
[spmatrix, tokenlist, trainCategory] = readMatrix(train_fname);


trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1);
numTokens = size(trainMatrix, 2);

% trainMatrix is now a (numTrainDocs x numTokens) matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents the number of times the j-th
% token appeared in email $i$. 

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% Note that for the SVM, you would want to convert these to +1 and -1.


% YOUR CODE HERE
total_spam = sum(trainCategory == 1);
total_nospam = sum(trainCategory == 0);

prob_spam = (total_spam + 1) / (numTrainDocs + 2);
prob_nospam = (total_nospam + 1) / (numTrainDocs + 2);

log_prob_spam = log(prob_spam);
log_prob_nospam = log(prob_nospam);

spam_word_freqs = sum(trainMatrix(trainCategory==1,:)) + 1;
nospam_word_freqs = sum(trainMatrix(trainCategory==0,:)) + 1;

spam_word_probs = spam_word_freqs / sum(spam_word_freqs);
nospam_word_probs = nospam_word_freqs / sum(nospam_word_freqs);

log_spam_word_probs = log(spam_word_probs);
log_nospam_word_probs = log(nospam_word_probs);
end