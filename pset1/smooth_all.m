load_quasar_data

[m,~] = size(train_qso);
train_qso_smooth = zeros(size(train_qso));
for i = 1:m
    train_qso_smooth(i,:) = smooth_f(train_qso(i,:), lambdas, 5);
end

[m,~] = size(test_qso);
test_qso_smooth = zeros(size(test_qso));
for i = 1:m
    test_qso_smooth(i,:) = smooth_f(test_qso(i,:), lambdas, 5);
end