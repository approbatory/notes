%------------------------------------------------------------
% ICA

load mix.dat	% load mixed sources
Fs = 11025; %sampling frequency being used

% listen to the mixed sources
normalizedMix = 0.99 * mix ./ (ones(size(mix,1),1)*max(abs(mix)));

% handle writing in both matlab and octave
%v = version;
%if (v(1) <= '3') % assume this is octave
%  wavwrite('mix1.wav', normalizedMix(:, 1), Fs, 16);
%  wavwrite('mix2.wav', normalizedMix(:, 2), Fs, 16);
%  wavwrite('mix3.wav', normalizedMix(:, 3), Fs, 16);
%  wavwrite('mix4.wav', normalizedMix(:, 4), Fs, 16);
%  wavwrite('mix5.wav', normalizedMix(:, 5), Fs, 16);
%else
%  wavwrite(normalizedMix(:, 1), Fs, 16, 'mix1.wav');
%  wavwrite(normalizedMix(:, 2), Fs, 16, 'mix2.wav');
%  wavwrite(normalizedMix(:, 3), Fs, 16, 'mix3.wav');
%  wavwrite(normalizedMix(:, 4), Fs, 16, 'mix4.wav');
%  wavwrite(normalizedMix(:, 5), Fs, 16, 'mix5.wav');
%end
audiowrite('mix1.wav', normalizedMix(:, 1), Fs);
audiowrite('mix2.wav', normalizedMix(:, 2), Fs);
audiowrite('mix3.wav', normalizedMix(:, 3), Fs);
audiowrite('mix4.wav', normalizedMix(:, 4), Fs);
audiowrite('mix5.wav', normalizedMix(:, 5), Fs);


W=eye(5);	% initialize unmixing matrix

% this is the annealing schedule I used for the learning rate.
% (We used stochastic gradient descent, where each value in the 
% array was used as the learning rate for one pass through the data.)
% Note: If this doesn't work for you, feel free to fiddle with learning
%  rates, etc. to make it work.
anneal = [0.1 0.1 0.1 0.05 0.05 0.05 0.02 0.02 0.01 0.01 ...
      0.005 0.005 0.002 0.002 0.001 0.001];
anneal = [anneal repmat(0.0005,1,50)];
obj = zeros(1,length(anneal));
sig = @(x) 1./(1+exp(-x));
sigp = @(x) sig(x).*(1-sig(x));
for iter=1:length(anneal)
   %%%% here comes your code part
   for i = randperm(size(normalizedMix,1))
       x = normalizedMix(i,:)';
       W = W + anneal(iter)*((1-2*sig(W*x))*x' + inv(W'));
   end
   obj(iter) = sum(sum(log(sigp(W*normalizedMix'))) + log(det(W)));
   fprintf('%d out of %d\tobj:%f\n', iter, length(anneal), obj(iter));
end
%%%% After finding W, use it to unmix the sources.  Place the unmixed sources 
%%%% in the matrix S (one source per column).  (Your code.) 

S = mix * W';

S=0.99 * S./(ones(size(mix,1),1)*max(abs(S))); 	% rescale each column to have maximum absolute value 1 

% now have a listen --- You should have the following five samples:
% * Godfather
% * Southpark
% * Beethoven 5th
% * Austin Powers
% * Matrix (the movie, not the linear algebra construct :-) 

%v = version;
%if (v(1) <= '3') % assume this is octave
%  wavwrite('unmix1.wav', S(:, 1), Fs, 16);
%  wavwrite('unmix2.wav', S(:, 2), Fs, 16);
%  wavwrite('unmix3.wav', S(:, 3), Fs, 16);
%  wavwrite('unmix4.wav', S(:, 4), Fs, 16);
%  wavwrite('unmix5.wav', S(:, 5), Fs, 16);
%else
%  wavwrite(S(:, 1), Fs, 16, 'unmix1.wav');
%  wavwrite(S(:, 2), Fs, 16, 'unmix2.wav');
%  wavwrite(S(:, 3), Fs, 16, 'unmix3.wav');
%  wavwrite(S(:, 4), Fs, 16, 'unmix4.wav');
%  wavwrite(S(:, 5), Fs, 16, 'unmix5.wav');
%end

audiowrite('unmix1.wav', S(:, 1), Fs);
audiowrite('unmix2.wav', S(:, 2), Fs);
audiowrite('unmix3.wav', S(:, 3), Fs);
audiowrite('unmix4.wav', S(:, 4), Fs);
audiowrite('unmix5.wav', S(:, 5), Fs);

%function res = sig(x)
%res = 1./(1+exp(-x));
%end
%
%function res = sigp(x)
%res = sig(x).*(1-sig(x));
%end