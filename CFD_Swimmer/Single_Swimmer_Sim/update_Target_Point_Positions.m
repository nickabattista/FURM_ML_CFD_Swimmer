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
% FUNCTION: updates the target point positions
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function targets = update_Target_Point_Positions(dt,current_time,targets,yPos,timeVec,DeltaT)


%IDs = targets(:,1);                 % Stores Lag-Pt IDs in col vector
%xPts= targets(:,2);                 % Previous x-Values of x-Target Pts.
%yPts= targets(:,3);                 % Previous y-Values of y-Target Pts.
%kStiffs = targets(:,4);             % Stores Target Stiffnesses 
%N_target = length(targets(:,1));    % Gives total number of target pts!


%---------------------------------------------
% KINEMATIC PARAMETERS
%---------------------------------------------
freq = 5;        % undulatory wave frequency
period = 1/freq; % undulatory wave period
numBody = 264;   % # of points in TOP (or bottom) of body individually!


%---------------------------------------------
% Find useful times in simulation
%---------------------------------------------
t = mod(current_time,period);     % mod by period   
tAux = mod(current_time,DeltaT);  % time in interpolation partition


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


%-----------------------------------------------------------
% Update the TARGET POINT POSITIONS -> only y-Position!
%-----------------------------------------------------------
targets(1:numBody,3) = yPos(1:numBody,ind_1) + (tAux/DeltaT)*( yPos(1:numBody,ind_2) - yPos(1:numBody,ind_1) );

