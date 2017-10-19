function f_smooth = smooth_f(f,lambdas,tau)
%SMOOTH_F Applies locally weighted linear regression to smooth f
% operates on f as a row vector
y = f.';
X = [ones(size(lambdas)), lambdas];

m = length(y);
n = 2;

thetas = zeros(n,m);

%Different weights per value of lambda
for i=1:m
    x = lambdas(i);
    ws = exp(-(x-lambdas).^2/(2*tau^2));
    W = 1/2*diag(ws);
    thetas(:,i) = (X.'*W*X)\(X.'*W*y);
end

%Evaluating the fit for each value of lambda
y_fit = zeros(1,m);
for i=1:m
    y_fit(i) = thetas(:,i).'*X(i,:).';
end

f_smooth = y_fit.';
end

