load_quasar_data
%first training example
y = train_qso(1,:).';
X = [ones(size(lambdas)), lambdas];

m = length(y);
n = 2;

thetas = zeros(n,m);
tau = 5;
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

%Plotting results
figure
plot(lambdas, y, '-x');
hold on
plot(lambdas, y_fit, 'r');
legend data fit
xlabel 'Wavelength \lambda'
ylabel 'Flux'
title 'weighted linear regression, \tau=5'