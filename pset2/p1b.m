clear;
dataA = load('data_a.txt');
dataB = load('data_b.txt');
nA = size(dataA, 1);
nB = size(dataB, 1);

XA = [ones(nA, 1), dataA(:, 2:end)];
XB = [ones(nB, 1), dataB(:, 2:end)];
yA = dataA(:,1);
yB = dataB(:,1);

sc(XA, yA, sprintf('dataset A, %d points', nA));
sc(XB, yB, sprintf('dataset B, %d points', nB));

function sc(X,y, titl)
figure
pos = y==1;
X1 = X(pos,:);
X0 = X(~pos,:);
scatter(X1(:,2), X1(:,3), 'x');
hold on
scatter(X0(:,2), X0(:,3), 'o');
title(titl)
end
