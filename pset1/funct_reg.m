%functional regression

%smooth_all %already ran
cutoff_lambda = 1300;
k = 3;
[m, ~] = size(train_qso_smooth);

fleft_trains = train_qso_smooth(:,lambdas < cutoff_lambda);
fright_trains = train_qso_smooth(:,lambdas >= cutoff_lambda);


fleft_tests = test_qso_smooth(:,lambdas < cutoff_lambda);
fright_tests = test_qso_smooth(:,lambdas >= cutoff_lambda);

disp('Average error for training set:');
disp(average_error(fleft_trains, fright_trains, fleft_trains, fright_trains, k));

disp('Average error for testing set:');
disp(average_error(fleft_tests, fright_tests, fleft_trains, fright_trains, k));

%plotting
for example = [1 6]
    figure
    plot(lambdas, test_qso_smooth(example,:));
    hold on
    fleft_hat = estimate_fleft(fright_tests(example,:), fleft_trains, fright_trains, k);
    plot(lambdas(lambdas < cutoff_lambda), fleft_hat);
    xlabel 'Wavelength \lambda'
    ylabel 'Flux'
    legend 'Spectrum f(\lambda)' 'Estimated f_{left}(\lambda)'
    title(sprintf('Example %d f_{left} estimation', example));
end

%The distance between two functions
function dd = d(f1, f2)
dd = sum((f1-f2).^2);
end

%The auxillary function defined in the problem set
function kk = ker(t)
kk = max(1-t,zeros(size(t)));
end

%Find the maximal distance from function f present in the training set
function hh = h(f, fright_trains)
[m, ~] = size(fright_trains);
hh = 0;
for i=1:m
    dist = d(f, fright_trains(i,:));
    if dist > hh
        hh = dist;
    end
end
end

%Find the k nearest neighbors (may include f) from f in the training set
function nn = neighb(k, f, fright_trains)
[m, ~] = size(fright_trains);
dists = zeros(m,1);
for i=1:m
    dists(i) = d(f, fright_trains(i,:));
end
[~, neighbor_inds] = sort(dists);
nn = neighbor_inds(1:k);
end

%Estimate fleft from fright by the weighted average from the k nearest
%neighbors where the weights are 1 - the distance to the neighbor divided
%by the maximal distance to any function in the training set
function fleft_hat = estimate_fleft(fright, fleft_trains, fright_trains, k)
nn = neighb(k, fright, fright_trains);
hh = h(fright, fright_trains);
numer = 0;
denom = 0;
for ind=nn.'
    fright_ind = fright_trains(ind,:);
    fleft_ind = fleft_trains(ind,:);
    numer = numer + fleft_ind * ker(d(fright, fright_ind)/hh);
    denom = denom + ker(d(fright, fright_ind)/hh);
end
fleft_hat = numer/denom;
end

%Estimate values for fleft from the frights and find the distance from the
%true fleft, then return the average such distance over the given set. This
%function can be used to find the average training error and the average
%testing error
function avgerr = average_error(flefts, frights, fleft_trains, fright_trains, k)
[m, ~] = size(flefts);
avgerr = 0;
for i=1:m
    fleft_hat = estimate_fleft(frights(i,:), fleft_trains, fright_trains, k);
    avgerr = avgerr + d(flefts(i,:), fleft_hat)/m;
end
end