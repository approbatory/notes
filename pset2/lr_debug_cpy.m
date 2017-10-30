% compute gradient given X, y, and theta
compute_grad = @(X, y, theta) (-1 / size(X, 1)) * X' * (y ./(1 + exp((X * theta).* y)));
compute_loss = @(X, y, theta) (1/size(X,1)) * sum(log1p(exp(-(X*theta).*y)));
%% training for dataset A
disp('============= Training model on dataset A ============');
data = load('data_a.txt');
n = size(data, 1);
% adding intercept to data X
X = [ones(n, 1), data(:, 2:end)];
y = data(:, 1);

% training a logistic model
theta = zeros(size(X, 2), 1);
learning_rate = 10;
%J = [];
%dgradnorm = [];
%saver = [];
for i = 1:10^9
    prev_theta = theta;
    grad = compute_grad(X, y, theta);
    theta = theta - learning_rate * grad;
    if mod(i, 10000) == 0
        fprintf('finished iteration %d \n', i);
    end
    %J = [J, compute_loss(X,y,theta)];
    %dgradnorm = [dgradnorm, norm(theta - prev_theta)];
    %saver = [saver, theta];
    if norm(theta - prev_theta) < 10^-15
        fprintf('converged in %d iterations \n', i-1);
        break;
    end
end
figure(1);
plot(X(y==1,2), X(y==1,3), 'x');
hold on
plot(X(y==-1,2), X(y==-1,3), 'o');
%t0 + t1x + t2y = 0
%t2y = -t1x - t0
%y = -t1/t2x - t0/t2
refline([-theta(2)/theta(3), -theta(1)/theta(3)]);
title('separation line for dataset A');
%figure(1);
%%plot(J);
%title('J(\theta) for dataset A');
%plot(saver');
%title('||\Delta Grad|| dataset A');
%title('parameters theta');
%% training for dataset B
disp('============= Training model on dataset B ============');
data = load('data_b.txt');
n = size(data, 1);
% adding intercept to data X
X = [ones(n, 1), data(:, 2:end)];
y = data(:, 1);

% training a logistic model
theta = zeros(size(X, 2), 1);
learning_rate = 10;
J = [];
%dgradnorm = [];
%saver = [];
for i = 1:10^9
    prev_theta = theta;
    grad = compute_grad(X, y, theta);
    theta = theta - learning_rate * grad;
    if mod(i, 10000) == 0
        fprintf('finished iteration %d \n', i);    
    end
    J = [J, compute_loss(X,y,theta)];
    %dgradnorm = [dgradnorm, norm(theta - prev_theta)];
    %saver = [saver, theta];
    if norm(theta - prev_theta) < 10^-15
        fprintf('converged in %d iterations \n', i-1);
        break;
    end
    if i > 1e5
        break;
    end
end
%%%figure(2);
%%%plot(X(y==1,2), X(y==1,3), 'x');
%%%hold on
%%%plot(X(y==-1,2), X(y==-1,3), 'o');
%t0 + t1x + t2y = 0
%t2y = -t1x - t0
%y = -t1/t2x - t0/t2
%%%refline([-theta(2)/theta(3), -theta(1)/theta(3)]);
%%%title('separation line for dataset B');
figure(2);
plot(J);
title('J(\theta) for dataset B');
%plot(saver');
%title('parameters theta');
%title('||\Delta Grad|| dataset B');