

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: main2d is the function that gets called to run the code. It
%           itself, reads in parameters from the input2d file, and passes 
%           them to the IBM_Driver function to run the simulation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function main2d()

% This is the "main" file, which gets called to initialized the Immersed Boundary
% Simulation. It reads in all the parameters from "input2d", and sends them
% off to the "IBM_Driver.m" function to actual perform the simulation

%
% Path Reference to where Driving code is found %
%
warning('off','all');

addpath('../IBM_Blackbox_REVISED_FEB2024/');


%
% THROW ERROR TO CORRECT PATH IF DRIVER IS NOT FOUND
%
assert( exist( 'IBM_Driver.m', 'file' ) == 2, 'IBM_Driver.m not found -> Please check path to IBM_Blackbox in main2d.m!' );


%
% READ-IN INPUT PARAMETERS %
%
[Fluid_Params, Grid_Params, Time_Params, Lag_Struct_Params, Output_Params, Lag_Name_Params,Con_Params] = please_Initialize_Simulation();


%-----------------------------------------------------------
%-%-%-% DO THE IMMERSED BOUNDARY SOLVE!!!!!!!! %-%-%-%
%-----------------------------------------------------------
%
% USES SPECIAL METACHRONAL FOR READING IN FILES EVERY TIME-STEP
%       IN UPDATE_TARGET_POINT_POSITIONS.M!!!!!
%
[X, Y, U, V, xLags, yLags,C] = IBM_Driver_SWIMMER(Fluid_Params,Grid_Params,Time_Params,Lag_Struct_Params,Output_Params,Lag_Name_Params,Con_Params);


%
% Print simulation has completed.
%
fprintf('\n\n');
fprintf(' |****** IMMERSED BOUNDARY SIMULATION HAS FINISHED! ******|\n\n')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to read in input files from "input2d" -> renames all
% quantities appropriately, just as they are in input file
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function [params,struct_name] = give_Me_input2d_Parameters()
% 
% filename= 'input2d';  %Name of file to read in
% 
% fileID = fopen(filename); %Opens file for 'textscan' function
%     % Read in the file, use 'CollectOutput' to gather all similar data together
%     % and 'CommentStyle' to to end and be able to skip lines in file.
%     C1 = textscan(fileID,'%s %s %f','CollectOutput',1,'CommentStyle','%');
%     C2 = textscan(fileID,'%s %s %s','CollectOutput',1,'CommentStyle','%');
% fclose(fileID);
% 
% params = C1{2};
% 
% struct_name = C2{1,1};

