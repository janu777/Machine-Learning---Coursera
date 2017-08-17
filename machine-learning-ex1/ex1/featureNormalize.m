function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
m=size(X,1);%number of training examples
n=size(X,2);
mu = zeros(1, size(X, 2));
K=ones(1,m);
sigma = zeros(1, size(X, 2));
L=ones(1,m);
N=X;
mu=(1/m)*(K*X);
for i=1:n
	for j=1:m
		N(j,i) = (X(j,i)-mu(i))^2;
end
end		 		
sigma=(1/(m-1))*(L*N);
sigma=sqrt(sigma);
for i=1:n
	for j=1:m
		X_norm(j,i) = (X(j,i)-mu(i))/sigma(i);
		
end
end
% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       









% ============================================================

end
