%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: applies data transformations to the model input parameters
%           so data has nearly zero averages and identical covariances
%   
%                           Transformations: 
%      Raw --> Linear Transform --> Power Transform --> Standardization
%
%   Author: Nick A Battista
%   Institution: The College of New Jersey
%   Date: April 2024
%       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [OUTPUTS_TRAIN,OUTPUTS_TEST,TRANSFORM_PARAMS] = scale_The_Output_Data(plot_FLAG,TRAINING_OUTPUTS,TESTING_OUTPUTS)

%------------------------------------------------------------------
%                     FLAG TO PLOT HISTOGRAMS
%------------------------------------------------------------------
plot_Histograms=plot_FLAG;
fs = 16; % Histogram FontSize 

%------------------------------------------------------------------
%      Combine ALL output data (both training and test datasets)
%------------------------------------------------------------------
SPEED = [TRAINING_OUTPUTS(:,1); TESTING_OUTPUTS(:,1)];
POWER = [TRAINING_OUTPUTS(:,2); TESTING_OUTPUTS(:,2)];

%------------------------------------------------------------------
%                     ORIGINAL DATA DISTRIBUTIONS 
%------------------------------------------------------------------
if plot_Histograms
    f1=figure(1);
    subplot(3,2,1)
    histogram(SPEED,25);
    xlabel('Speed (Raw)'); ylabel('Frequency');
    set(gca,'FontSize',fs);    
    grid on;
    grid minor;
    %
    subplot(3,2,2)
    histogram(POWER,25);
    xlabel('Power (Raw)'); ylabel('Frequency');    
    set(gca,'FontSize',fs);    
    grid on;
    grid minor;
end

%------------------------------------------------------------------
%                     LINEAR TRANSFORMATION 
%   (to make sure data is non-zero before BOX-COX/Power Transform)
%------------------------------------------------------------------
addS = 0.0;
addP = 0.0;
%
SPEED =                                 SPEED + addS;
TRAINING_OUTPUTS(:,1) = TRAINING_OUTPUTS(:,1) + addS;
TESTING_OUTPUTS(:,1) =   TESTING_OUTPUTS(:,1) + addS;
%
POWER =                                 POWER + addP;
TRAINING_OUTPUTS(:,2) = TRAINING_OUTPUTS(:,2) + addP;
TESTING_OUTPUTS(:,2) =   TESTING_OUTPUTS(:,2) + addP;

%------------------------------------------------------------------
%                      BOX-COX TRANSFORMATION
% POWER TRANSFORM: https://en.wikipedia.org/wiki/Power_transform
%------------------------------------------------------------------
lam_S = 0.4;    % Power transform parameter (for speed)
lam_P = 0.125;  % Power transform parameter (for power)
%
GM_S = geomean(SPEED);
SPEED =                                 ( SPEED.^(lam_S) - 1 ) / ( lam_S * GM_S^(lam_S-1) );
TRAINING_OUTPUTS(:,1) = ( TRAINING_OUTPUTS(:,1).^(lam_S) - 1 ) / ( lam_S * GM_S^(lam_S-1) );
TESTING_OUTPUTS(:,1) =   ( TESTING_OUTPUTS(:,1).^(lam_S) - 1 ) / ( lam_S * GM_S^(lam_S-1) );
%
GM_P = geomean(POWER);
POWER =                                 ( POWER.^(lam_P) - 1 ) / ( lam_P * GM_P^(lam_P-1) );
TRAINING_OUTPUTS(:,2) = ( TRAINING_OUTPUTS(:,2).^(lam_P) - 1 ) / ( lam_P * GM_P^(lam_P-1) );
TESTING_OUTPUTS(:,2) =   ( TESTING_OUTPUTS(:,2).^(lam_P) - 1 ) / ( lam_P * GM_P^(lam_P-1) );
%

%------------------------------------------------------------------
% Calculate Average and Variance for each Quantity of Interest
%------------------------------------------------------------------
avgS = mean(SPEED);
varS =  var(SPEED);
%
avgP = mean(POWER);
varP = var(POWER);


%------------------------------------------------------------------
%              DATA DISTRIBUTIONS AFTER POWER TRANSFORM
%------------------------------------------------------------------
if plot_Histograms
    subplot(3,2,3)
    histogram(SPEED,25);
    xlabel('Speed (after Power Transform)'); ylabel('Frequency');    
    set(gca,'FontSize',fs);    
    grid on;
    grid minor;
    %
    subplot(3,2,4)
    histogram(POWER,25);
    xlabel('Power (after Power Transform)'); ylabel('Frequency');    
    set(gca,'FontSize',fs);    
    grid on;
    grid minor;
end


%------------------------------------------------------------------
%              STANDARDIZATION (z-Scrore transformation)
%------------------------------------------------------------------
TRAINING_OUTPUTS(:,1) = ( TRAINING_OUTPUTS(:,1) - avgS ) / sqrt(varS);
TESTING_OUTPUTS(:,1) =  ( TESTING_OUTPUTS(:,1)  - avgS ) / sqrt(varS);
%
TRAINING_OUTPUTS(:,2) = ( TRAINING_OUTPUTS(:,2) - avgP ) / sqrt(varP);
TESTING_OUTPUTS(:,2) =  ( TESTING_OUTPUTS(:,2)  - avgP ) / sqrt(varP);


%------------------------------------------------------------------
%            DATA DISTRIBUTIONS AFTER STANDARDIZATION
%------------------------------------------------------------------
if plot_Histograms
    subplot(3,2,5)
    histogram([TRAINING_OUTPUTS(:,1)' TESTING_OUTPUTS(:,1)'],25);
    xlabel('Speed (after Standardization)'); ylabel('Frequency');    
    set(gca,'FontSize',fs);    
    grid on;
    grid minor;
    %
    subplot(3,2,6)
    histogram([TRAINING_OUTPUTS(:,2)' TESTING_OUTPUTS(:,2)'],25);
    xlabel('Power (after Standardization)'); ylabel('Frequency'); 
    set(gca,'FontSize',fs);    
    grid on;
    grid minor;
    %
    f1.Position=[25 25 1100 800];
    pause(2);
end


%------------------------------------------------------------------
%    STORE TRANSFORMED OUTPUTS AND TRANSFORMATION PARAMETERS
%------------------------------------------------------------------
OUTPUTS_TRAIN = [TRAINING_OUTPUTS(:,1) TRAINING_OUTPUTS(:,2)];
OUTPUTS_TEST  = [TESTING_OUTPUTS(:,1)  TESTING_OUTPUTS(:,2)];

PARAMS_S = [addS lam_S GM_S avgS varS]';
PARAMS_P = [addP lam_P GM_P avgP varP]';

TRANSFORM_PARAMS(:,1) = PARAMS_S;
TRANSFORM_PARAMS(:,2) = PARAMS_P;





