load_quasar_data
%first training example
y = train_qso(1,:).';
X = [ones(size(lambdas)), lambdas];

m = length(y);
n = 2;

thetas = zeros(n,m);
for tau = [1 10 100 1000]
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

figure
plot(lambdas, y, '-x');
hold on
plot(lambdas, y_fit);
legend_message = sprintf('\\tau=%d', tau);
legend('data', legend_message); 
xlabel 'Wavelength \lambda'
ylabel 'Flux'
title(sprintf('weighted linear regression, %s', legend_message));
end