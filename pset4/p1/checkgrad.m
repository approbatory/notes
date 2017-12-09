% initialize the parameters
W1 = randn(h1, n);
b1 = zeros(h1, 1);
W2 = randn(c, h1);
b2 = zeros(c, 1);

Db2 = 1e-3*ones(size(b2));
[~,~,loss1] = forward_prop(X_train, y_train, W1, b1, W2, b2, 0);
[~,~,loss2] = forward_prop(X_train, y_train, W1, b1, W2, b2 + Db2, 0);
[~, ~, ~, db2] = backward_prop(X_train, y_train, W1, b1, W2, b2, 0);
disp(loss2 - loss1);
disp(Db2(:)'*db2(:));


DW1 = 1e-3*ones(size(W1));
[~,~,loss1] = forward_prop(X_train, y_train, W1, b1, W2, b2, 0);
[~,~,loss2] = forward_prop(X_train, y_train, W1+DW1, b1, W2, b2, 0);
[dW1, ~, ~, ~] = backward_prop(X_train, y_train, W1, b1, W2, b2, 0);
disp(loss2 - loss1);
disp(DW1(:)'*dW1(:));