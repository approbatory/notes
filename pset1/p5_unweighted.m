load_quasar_data
%first training example
y = train_qso(1,:).';
X = [ones(size(lambdas)), lambdas];
theta = (X.'*X)\(X.'*y);
disp(theta);

%Plotting results
figure
plot(lambdas, y, '-x');
hold on;
ref = refline(flipud(theta));
ref.Color = 'r';
legend data fit
xlabel 'Wavelength \lambda'
ylabel 'Flux'
title 'unweighted linear regression'