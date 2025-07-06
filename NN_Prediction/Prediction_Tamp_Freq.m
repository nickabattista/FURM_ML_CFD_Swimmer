%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: Uses the NN to predict performance across the (Tamp,freq)-subspace
%           for constant values of Hamp/Tamp, lambda, and xB*
%
%   Author: Nick A Battista
%   Institution: The College of New Jersey
%   Date: April 2024
%       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Prediction_Tamp_Freq()

%----------------------------------------------------
%           5-D PARAMETER SPACE (in order)
% 
% TAIL AMPLITUDE (Tamp): [0.05 0.30]
% HEAD/TAIL RATIO (Hamp/Tamp): [0.0,0.80]
% WAVELENGTH (lambda): [0.5,1.5]
% FREQUENCY (f): [1,5]
% ENVELOPE BODY PT (xB*): [0,1]
%----------------------------------------------------
%
TailAmpVEC=linspace(0.05,0.30,10); 
FreqVEC = linspace(1,5,10);
%
HampTampRatioVAL = 0.2; 
WavelengthVAL= 0.925;
EnvelopeBodyVAL= 0.95;


%----------------------------------------------------
%        LOAD NN's WEIGHT AND BIAS MATRICES 
%          (*after NN has been trained*)
%----------------------------------------------------
%load('../NN_Training/Trained_Neural_Network_Weights_PAPER/Trained_Neural_Network.mat');
load('../NN_Training/Trained_Neural_Network_Weights/Trained_Neural_Network.mat');

%----------------------------------------------------
%     LOAD Data Transformation Information
%----------------------------------------------------
load('../NN_Training/DATA_TRANSFORM_INFO.mat');


%----------------------------------------------------
%      ADD PATH TO NN FORWARD PROPAGATION SCRIPT
%----------------------------------------------------
NN_Path = '../NN_Training/';
addpath(NN_Path);


%------------------------------------------------------
% CONSTRUCT MESH FOR PLOTTING AND ALLOCATE STORAGE
%------------------------------------------------------
xData = TailAmpVEC;
yData = FreqVEC;
Z1 = zeros( length(yData), length(xData) );
Z2 = zeros( length(yData), length(xData) );
%-----------------------------------------------------------------------------
% LOOP THRU ALL COMBINATIONS IN SUBSPACE AND USE NN TO PREDICT PERFORMANCE
%-----------------------------------------------------------------------------
for i=1:length(xData)
    for j=1:length(yData)
                
        %-------------------------------------------
        %   Select Specific Parameter Combination
        %-------------------------------------------
        INPUT_VEC = [TailAmpVEC(i) HampTampRatioVAL WavelengthVAL FreqVEC(j) EnvelopeBodyVAL];
                
        %-------------------------------------------
        %  Scale Specific Parameters Appropriately
        %-------------------------------------------
        INPUT_VEC_SCALED = scale_Input_Parameters_for_NN(INPUT_VEC);
        
        %-------------------------------------------
        %        NEURAL NETWORK PREDITION....
        %-------------------------------------------
        outputVec = forward_propagate(INPUT_VEC_SCALED',W1,W2,WEnd,b1,b2);
        Z1(j,i) = outputVec(1); % Speed
        Z2(j,i) = outputVec(2); % Power

    end
end



%--------------------------------------
%     REMOVE PATH TO NN SCRIPTS
%--------------------------------------
rmpath(NN_Path);


%----------------------------------------------------
%           DATA TRANSFORMATION PARAMETERS
%----------------------------------------------------
addS = TRANSFORM_PARAMS(1,1); addP = TRANSFORM_PARAMS(1,2);
lamS = TRANSFORM_PARAMS(2,1); lamP = TRANSFORM_PARAMS(2,2);
GM_S = TRANSFORM_PARAMS(3,1); GM_P = TRANSFORM_PARAMS(3,2);
avgS = TRANSFORM_PARAMS(4,1); avgP = TRANSFORM_PARAMS(4,2);
varS = TRANSFORM_PARAMS(5,1); varP = TRANSFORM_PARAMS(5,2);

%----------------------------------------------------
%           UNDO 'z-Score' transformation
%----------------------------------------------------
Z1 = sqrt(varS) * Z1 + avgS;
Z2 = sqrt(varP) * Z2 + avgP;


%----------------------------------------------------
%             UNDO 'Power' transformation
%----------------------------------------------------
Z1 = ( lamS * GM_S^(lamS-1) * Z1 + 1 ).^(1/lamS);
Z2 = ( lamP * GM_P^(lamP-1) * Z2 + 1 ).^(1/lamP);
 
%----------------------------------------------------
%           UNDO 'Linear' transformation
%   (afterward, Z1 and Z2 in raw simulation units)
%----------------------------------------------------
Z1 = Z1 - addS;
Z2 = Z2 - addP;


%--------------------------------------------------------------------
%
%                             FIGURES:
%
%               UNDULATION FREQUENCY IS ON HORIZONTAL
%             (Each curve is different Tail Amplitude)
%
%--------------------------------------------------------------------

%----------------------------------------------
%               Plot Attributes
%----------------------------------------------
ms = 35;  % MarkerSize
lw = 6;   % LineWidth
fs = 22;  % FontSize
xLIM = [min(yData)-0.05*max(yData) 1.05*max(yData)];% FREQUENCY ON HORIZONTAL
yLIM_Speed = [0 max(max(Z1))+0.25];                 % y-Axis limits (Speed)
yLIM_Power = [0.9*min(min(Z2)) 1.1*max(max(Z2))];   % y-Axis limits (Power)

%----------------------------------------------
%  Down sample colormap to # of tail amplitudes
%----------------------------------------------
len = length(xData);          % # of tail amplitudes to loop through (Freq, horizontal)
cMAT=colormap(parula);
cInds = floor(linspace(1,64,len));
cMAT = cMAT(cInds,:);

%----------------------------------------------
%              FIGURE 1: SPEED
%----------------------------------------------
f1 = figure(1);
%
legVec = [];  % To store legend information
%
for i=1:length(xData)
    plot(yData,Z1(:,i),'.-','MarkerSize',ms','LineWidth',lw,'Color',cMAT(i,:) ); hold on;
    legVec{i} = ['Tamp=' num2str(xData(i),'%.2f')];
end
leg1 = legend(legVec);
xlabel('Undulation Frequency');
ylabel('Speed'); 
set(gca,'FontSize',fs);
set(leg1,'FontSize',fs-4,'Location','NorthWest');
leg1.NumColumns=2;
grid on;
grid minor;
xlim(xLIM);
ylim(yLIM_Speed);

%----------------------------------------------
%              FIGURE 2: POWER
%----------------------------------------------
f2 = figure(2);
%
legVec = [];  % To store legend information
%
for i=1:length(xData)
    semilogy(yData,Z2(:,i),'.-','MarkerSize',ms','LineWidth',lw,'Color',cMAT(i,:) ); hold on;
    legVec{i} = ['Tamp=' num2str(xData(i),'%.2f')];
end
leg2 = legend(legVec);
xlabel('Undulation Frequency');
ylabel('Power'); 
set(gca,'FontSize',fs);
set(leg2,'FontSize',fs-4,'Location','SouthEast');
leg2.NumColumns=2;
grid on; 
grid minor;
xlim(xLIM);
ylim(yLIM_Power);

f1.Position = [ 10 500 650 400];
f2.Position = [660 500 650 400];



