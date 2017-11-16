A_large = double(imread('mandrill-large.tiff'));
%imshow(uint8(round(A_large)));


%%
A_small = double(imread('mandrill-small.tiff'));
%imshow(uint8(round(A_small)));

%%
n_channels = 3;
C = reshape(A_small,[],n_channels);
n_pix = size(C,1);

K = 16;
mu = C(randi(n_pix,K,1),:);
class = zeros(n_pix,1);
T = 1000;
J = zeros(T,1);
for iter = 1:T
    class_old = class(:);
    for i = 1:n_pix
        min_dist = Inf;
        for j = 1:K
            d = norm(C(i,:) - mu(j,:));
            if d < min_dist
                min_dist = d;
                class(i) = j;
            end
        end
    end
    for j = 1:K
        mu(j,:) = mean(C(class==j,:));
    end
    J(iter) = mean(sum((C - mu(class,:)).^2,2));
    if all(class_old == class)
        disp('Converged');
        break;
    end
end
J = J(1:iter);
%%
[Nx, Ny, n_channels] = size(A_large);
A_compressed = zeros(size(A_large));
for x = 1:Nx
    for y = 1:Ny
        color = reshape(A_large(x,y,:),1,[]);
        min_dist = Inf;
        for j = 1:K
            d = norm(color - mu(j,:));
            if d < min_dist
                min_dist = d;
                A_compressed(x,y,:) = mu(j,:);
            end
        end
    end
end
imshow(uint8(round(A_compressed)));
imwrite(uint8(round(A_compressed)), 'kmeans_image.tiff');