%Load the data into arrays
X = load('logistic_x.txt');
Y = load('logistic_y.txt');

%Using N=2 but N+1 for the size of theta to account for an intercept term
N = 2;
M = length(X);

%Put in constant one feature into X
X = [ones(M,1), X];

%Initialize theta to be all zeros
theta = zeros(N+1,1);

%Allow sufficient amount of time for convergence
T=10;
for t=1:T
    H = 0;
    grad = 0;
    %Calculate values for hessian 'H' and gradient 'grad' using components
    %from each data point (x^(i), y^(i))
    for i = 1:M
        xi = X(i,:).';
        yi = Y(i);
        expo = exp(yi*theta.'*xi);
        H = H + expo/(1+expo)^2 * xi*(xi.')/M;
        grad = grad + (-yi*xi)/(1+expo)/M;
    end
    disp(t);
    disp(grad.');
    %Apply Netwon's method to theta
    theta = theta - H\grad;
end

%Plotting results
plot(X(Y==-1,2), X(Y==-1,3), 'o');
hold on
plot(X(Y==1,2), X(Y==1,3), 'x');
bound = refline(-theta(2)/theta(3), -theta(1)/theta(3));
bound.Color = 'k';
legend('-1', '+1', 'bound.', 'Location', 'SouthEast');
xlabel x_1
ylabel x_2
title 'logistic regression'
display(theta);