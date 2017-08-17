function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
g=X*theta;
h=sigmoid(g);
R1=theta;
R1(1)=0;
cost_function=(y'*log(h))+((1-y)'*log(1-h));
% You need to return the following variables correctly 
J = 0;
J = (-(1/m)*cost_function)+((lambda/(2*m))*(sum(R1.^2)));
grad = zeros(size(theta));
R=(lambda/m)*theta;
R(1)=0;
grad=((1/m)*(X'*(h-y)))+R;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% =============================================================

end
