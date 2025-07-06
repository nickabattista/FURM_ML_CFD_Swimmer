%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: perform forward propagation
%           
%       Step: [1] x1 = W1*x0 + b1  (multiply by WEIGHT-1 matrix + add bias)
%             [2] A1 = A(x1)       (apply activation function to x1)
%             [3] x2 = W2*A1 + b2  (multiply by WEIGHT-2 matrix + add bias)
%             [4] A2 = A(x2)       (apply activation function to x2)
%             [5] zHat = WEnd*A2   (multiply by WEIGHT-End matrix)
%
%   Author: Nick A Battista
%   Institution: The College of New Jersey
%   Date: April 2024
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%s

function [zHat,x2,x1] = forward_propagate(x0,W1,W2,WEnd,b1,b2)

%----------------------------------------
% Compute Z1 = W1*x0
%           (1ST HIDDEN LAYER)
%----------------------------------------
x1 = W1*x0 + b1;

%----------------------------------------
% Apply activation function to x1
%       (redefine x1 in the process)
%----------------------------------------
x1 = activation_function( x1 );

%----------------------------------------
% Compute Z2 = W2*x1
%           (2ND HIDDEN LAYER)
%----------------------------------------
x2 = W2*x1 + b2;

%----------------------------------------
% Apply activation function to x2
%       (redefine x2 in the process)
%----------------------------------------
x2 = activation_function( x2 );

%---------------------------------------------
% Compute zHat = WEnd*x1
%       (NO activation function here)
%---------------------------------------------
zHat = WEnd*x2;


