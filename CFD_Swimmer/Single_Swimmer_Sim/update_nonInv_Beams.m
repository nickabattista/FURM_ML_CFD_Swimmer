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
% FUNCTION: updates the beam attributes!
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function beams_info = update_nonInv_Beams(dt,current_time,beams_info,xPos,yPos,timeVec,DeltaT)


% beams_info:   col 1: 1ST PT.
%               col 2: MIDDLE PT. (where force is exerted)
%               col 3: 3RD PT.
%               col 4: beam stiffness
%               col 5: x-curvature
%               col 6: y-curvature

%IDs = beams_info(:,1);   % Gives Middle Pt.

%---------------------------------------------
% KINEMATIC PARAMETERS
%---------------------------------------------
freq = 5;         % undulatory wave frequency
period = 1/freq;  % undulatory wave period


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



%-------------------------------------------------
% Update the LAGRANGIAN POINT POSITIONS
%-------------------------------------------------
xPts = xPos(:,ind_1) + (tAux/DeltaT)*( xPos(:,ind_2) - xPos(:,ind_1) );
yPts = yPos(:,ind_1) + (tAux/DeltaT)*( yPos(:,ind_2) - yPos(:,ind_1) );


%-------------------------------------------------
% NOW WE UPDATE THE CURAVTURES APPROPRIATELY
%-------------------------------------------------
beams_info(:,5) = xPts( beams_info(:,1) )+xPts( beams_info(:,3) )-2*xPts( beams_info(:,2) );
beams_info(:,6) = yPts( beams_info(:,1) )+yPts( beams_info(:,3) )-2*yPts( beams_info(:,2) );

