%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: Choose an Activation function
%           
%       NOTE: function expects input as matrix (or vector)
%
%   Author: Nick A Battista
%   Institution: The College of New Jersey
%   Date: April 2024
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = activation_function( Z )

%--------------------------------
% SIGMOID ACTIVATION
%--------------------------------
%A = 1 ./ ( 1 + exp( -Z ) );

%--------------------------------
% ReLU
%--------------------------------
A=max(0,Z);

