%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: DERIVATIVE of the SELECTED activation function
%           
%       NOTE: function expects input as matrix (or vector)
%
%   Author: Nick A Battista
%   Institution: The College of New Jersey
%   Date: April 2024
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = act_function_PRIME( Z )

%--------------------------------
% SIGMOID DERIVATIVE
%--------------------------------
%A = exp( -Z ) ./ ( 1 + exp( -Z ) ).^2;

%--------------------------------
% ReLU DERIVATIVE
%--------------------------------
A = zeros(size(Z));
%
A( find(Z>0) ) = 1;


