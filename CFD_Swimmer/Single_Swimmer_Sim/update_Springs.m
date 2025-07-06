%------------------------------------------------------------------------------------------%
%
% IB2d is an Immersed Boundary Code (IB) for solving fully coupled non-linear 
% 	fluid-structure interaction models. This version of the code is based off of
%	Peskin's Immersed Boundary Method Paper in Acta Numerica, 2002.
%
% Author: Nicholas A. Battista
% Email:  battistn[@]tcnj[.]edu
% 
% IB2d was Created: May 27th, 2015 at UNC-CH
%
% This code is capable of creating Lagrangian Structures using:
% 	1. Springs
% 	2. Beams (*torsional springs)
% 	3. Target Points
% 	4. Muscle-Model (combined Force-Length-Velocity model, "Hill+(Length-Tension)")
%   .
%   .
% 
% There are a number of built in Examples, mostly used for teaching purposes. 
%
%------------------------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: updates the spring attributes!
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function springs_info = update_Springs(dt,current_time,xLag,yLag,springs_info,xPos,yPos,timeVec,DeltaT)

%springs_info: col 1: starting spring pt (by lag. discretization)
%              col 2: ending spring pt. (by lag. discretization)
%              col 3: spring stiffness
%              col 4: spring resting lengths

%RL = springs_info(:,4); % resting-length vector

%---------------------------------------------
% KINEMATIC PARAMETERS
%---------------------------------------------
freq = 5;        % undulatory wave frequency
period = 1/freq; % undulatory wave period
numBody = 264;   % # of points in TOP (or bottom) of body individually!

%---------------------------------------------
% Find useful times in simulation
%---------------------------------------------
t = mod(current_time,period);      % mod by period   
tAux = mod(current_time,DeltaT);   % time in interpolation partition


%---------------------------------------------
% Find which section we're interpolating btwn
%---------------------------------------------
notFound = 1;
ct=0;
while notFound
    ct = ct+1;
    if t < timeVec(ct)
        ind_1 = ct-1;
        ind_2 = ct;
        notFound = 0;
    end
end

%------------------------------------------------------------
% Update the BODY LAGRANGIAN POINT POSITIONS 
%       in order to update resting lengths appropriately
%------------------------------------------------------------
% FOR TOP UNDULATORY BODY
xPts1 = xPos(1:numBody,ind_1) + (tAux/DeltaT)*( xPos(1:numBody,ind_2) - xPos(1:numBody,ind_1) );
yPts1 = yPos(1:numBody,ind_1) + (tAux/DeltaT)*( yPos(1:numBody,ind_2) - yPos(1:numBody,ind_1) );


%--------------------------------------------------------
% NOW WE CALCULATE DISTANCES BETWEEN SUCCESSIVE SPRINGS!
%--------------------------------------------------------
distVec1 = sqrt( ( xPts1(2:end) - xPts1(1:end-1) ).^2 + ( yPts1(2:end) - yPts1(1:end-1) ).^2 );

%-------------------------------------------------
% Update Resting Lengths!
%-------------------------------------------------
springs_info(:,4) = distVec1;





