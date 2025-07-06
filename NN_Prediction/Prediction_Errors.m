%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: Checks how well the trained NN recovers the training and 
%           testing datasets
%
%   Author: Nick A Battista
%   Institution: The College of New Jersey
%   Date: April 2024
%       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Prediction_Errors()


MG = 1.2*[205 27 125]/255;  % Magenta
DB = [0.4 0.5 0.95];        % Default Blue
BG = 1.1*[0.25 0.75 0.80];  % Blue-Green
RO = 0.8*[0.98 0.35 0.25];  % Red-Orange


%----------------------------------------------------
%        LOAD NN's WEIGHT AND BIAS MATRICES 
%          (*after NN has been trained*)
%----------------------------------------------------
load('../NN_Training/Trained_Neural_Network_Weights/Trained_Neural_Network.mat');



%----------------------------------------------------
%      ADD PATH TO NN FORWARD PROPAGATION SCRIPT
%----------------------------------------------------
NN_Path = '../NN_Training/';
addpath(NN_Path);



%---------------------------------------------------------------------------
%                   LOAD THE TRAINING AND TESTING DATA
%        
%      --> TRAINING_INPUTS: 1000x5 matrix of model inputs (real-values)
%      --> TESTING_INPUTS:  2000x5 matrix of model inputs (real-values)
%               ORDER: [Tamp, Hamp/Tamp, lambda, freq, xB*]
%
%      --> TRAINING_OUTPUTS: 1000x2 matrix of model outputs 
%      --> TESTING_OUTPUTS:  2000x2 matrix of model outputs 
%               ORDER: col-1: speed (raw), col-2: power (raw)
%        
%---------------------------------------------------------------------------
load('../NN_Training/Training_and_Test_Datasets_PAPER.mat');
%
numInputs = length(TRAINING_INPUTS(1,:));        % # of model inputs
numOutputs = length(TRAINING_OUTPUTS(1,:));      % # of model outputs
numSamplesTrain = length(TRAINING_INPUTS(:,1));  % # of training samples
numSamplesTest = length(TESTING_INPUTS(:,1));    % # of testing samples



%---------------------------------------------------------------------------
%                 LOAD DATA TRANSFORMATIONS PARAMETERS:
%       (from Linear --> Raw --> Power Transform --> Standardization)
%   
%   LOADS:
%   {TRAINING_OUTPUTS_TRANSFORM,TESTING_OUTPUTS_TRANSFORM,TRANSFORM_PARAMS'}
%
%   USE TO TRANSFORM NN PREDICTIONS BACK TO REAL VALUED OUTPUT SPACES!
%---------------------------------------------------------------------------
load('../NN_Training/DATA_TRANSFORM_INFO.mat');


%---------------------------------------------------------------------------
%               SCALE MODEL INPUTS (PARAMETERS) TO [-1,1]
%                  from [real min, real max] --> [-1,1]
%---------------------------------------------------------------------------
[TRAINING_INPUTS_SCALED,TESTING_INPUTS_SCALED] = scale_Training_and_Testing_Data(numInputs,TRAINING_INPUTS,TESTING_INPUTS);



%--------------------------------------------------------------
%                 MODEL TRAINING PREDICTIONS...
%   --> Perform Forward Propagation on Training Model Inputs 
%       (to get NN model predicted values for comparison)
%--------------------------------------------------------------
[zTrain_Transformed,~,~] = forward_propagate(TRAINING_INPUTS_SCALED',W1,W2,WEnd,b1,b2);
zTrain_Transformed = zTrain_Transformed'; % Transpose!
 

%--------------------------------------------------------------
%                   MODEL TESTING PREDICTIONs....
%   --> Perform Forward Propagation on Testing Model Inputs 
%       (to get NN model predicted values for comparison)
%--------------------------------------------------------------
[zTest_Transformed,~,~] = forward_propagate(TESTING_INPUTS_SCALED',W1,W2,WEnd,b1,b2);
zTest_Transformed = zTest_Transformed'; % Transpose!


%--------------------------------------
%     REMOVE PATH TO NN SCRIPTS
%--------------------------------------
rmpath(NN_Path);


%-----------------------------------------------------
%           DATA TRANSFORMATION PARAMETERS
%-----------------------------------------------------
addS = TRANSFORM_PARAMS(1,1); addP = TRANSFORM_PARAMS(1,2);
lamS = TRANSFORM_PARAMS(2,1); lamP = TRANSFORM_PARAMS(2,2);
GM_S = TRANSFORM_PARAMS(3,1); GM_P = TRANSFORM_PARAMS(3,2);
avgS = TRANSFORM_PARAMS(4,1); avgP = TRANSFORM_PARAMS(4,2);
varS = TRANSFORM_PARAMS(5,1); varP = TRANSFORM_PARAMS(5,2);

%-----------------------------------------------
%          UNDO 'z-Score' transformation
%-----------------------------------------------
sHat_1 = sqrt(varS) * zTrain_Transformed(:,1) + avgS;
sHat_2 = sqrt(varS) *  zTest_Transformed(:,1) + avgS;

pHat_1 = sqrt(varP) * zTrain_Transformed(:,2) + avgP;
pHat_2 = sqrt(varP) *  zTest_Transformed(:,2) + avgP;

%-----------------------------------------------
%          UNDO 'Power' transformation
%-----------------------------------------------
sHat_1 = ( lamS * GM_S^(lamS-1) * sHat_1 + 1 ).^(1/lamS);
sHat_2 = ( lamS * GM_S^(lamS-1) * sHat_2 + 1 ).^(1/lamS);

pHat_1 = ( lamP * GM_P^(lamP-1) * pHat_1 + 1 ).^(1/lamP);
pHat_2 = ( lamP * GM_P^(lamP-1) * pHat_2 + 1 ).^(1/lamP);

%-----------------------------------------------
%         UNDO 'Linear' transformation
%-----------------------------------------------
zTrain_Transformed(:,1) = sHat_1 - addS;
 zTest_Transformed(:,1) = sHat_2 - addS;
%
zTrain_Transformed(:,2) = pHat_1 - addP;
 zTest_Transformed(:,2) = pHat_2 - addP;
 
 
%------------------------------------------------------------
%
% PRINT ERROR INFO (both TRAINING/VALIDATION & TEST DATA)
%
%------------------------------------------------------------
%
% MODEL VALIDATION (how well the NN recovers the TRAINING dataset)
print_Error_Information(zTrain_Transformed,TRAINING_OUTPUTS,'VALIDATION',numOutputs);
%
% MODEL TESTING (how well the NN predicts the TEST dataset)
print_Error_Information(zTest_Transformed,TESTING_OUTPUTS,'TESTING',numOutputs);





%-----------------------------------------------------------------------------
%                    MODEL VALIDATION AND TESTING PLOTS (SPEED): 
%       --> compare true values against NN Predicted values!
%       --> "see how well NN recovered the TRAINING dataset"
%       --> "see how well NN predicted the TEST dataset"
%-----------------------------------------------------------------------------
BLUEGREEN = [0 208 200]/255;
BLACK = [0 0 0];
ms = 30;   % MarkerSize
lw=5;      % LineWidth
fs=22;     % FontSize
%
f3 = figure(3);
subplot(2,1,1)
plot(1:numSamplesTrain, TRAINING_OUTPUTS(:,1),'.','MarkerSize',ms+15,'Color',BLACK); hold on;
plot(1:numSamplesTrain, zTrain_Transformed(:,1),'.','MarkerSize',ms+5,'Color',BLUEGREEN); hold on;
leg=legend('Training Data','ANN Prediction (Validation)');
xlabel('Training Sample ID');
ylabel('Speed');
axis([0 numSamplesTrain+1 0 1.05*max(TRAINING_OUTPUTS(:,1))]);
title('NN Model: Validation');
set(gca,'FontSize',fs);
set(leg,'FontSize',18);
grid on; 
grid minor;
%
subplot(2,1,2)
plot(1:numSamplesTest, TESTING_OUTPUTS(:,1),'.','MarkerSize',ms+20,'Color',BLACK); hold on;
plot(1:numSamplesTest, zTest_Transformed(:,1),'.','MarkerSize',ms+5,'Color',BLUEGREEN); hold on;
leg=legend('Testing Data','NN Prediction (Testing)');
xlabel('Testing Sample ID');
ylabel('Speed');
axis([0 numSamplesTest+1 0 1.05*max(TESTING_OUTPUTS(:,1))]);
title('NN Model: Testing');
set(gca,'FontSize',fs);
set(leg,'FontSize',18);
grid on;
grid minor;
%
f3.Position = [100 120 1400 800];


%-----------------------------------------------------------------------------
%                    MODEL VALIDATION AND TESTING PLOTS (POWER): 
%       --> compare true values against NN Predicted values!
%       --> "see how well NN recovered the TRAINING dataset"
%       --> "see how well NN predicted the TEST dataset"
%-----------------------------------------------------------------------------
f4 = figure(4);
subplot(2,1,1)
semilogy(1:numSamplesTrain, TRAINING_OUTPUTS(:,2),'.','MarkerSize',ms+20,'Color',BLACK); hold on;
semilogy(1:numSamplesTrain, zTrain_Transformed(:,2),'.','MarkerSize',ms+5,'Color',BLUEGREEN); hold on;
leg=legend('Training Data','ANN Prediction (Validation)');
xlabel('Training Sample ID');
ylabel('Power');
axis([0 numSamplesTrain+1 0 1.05*max(TRAINING_OUTPUTS(:,2))]);
title('NN Model: Validation');
set(gca,'FontSize',fs);
set(leg,'FontSize',18);
grid on; 
grid minor;
%
subplot(2,1,2)
semilogy(1:numSamplesTest, TESTING_OUTPUTS(:,2),'.','MarkerSize',ms+20,'Color',BLACK); hold on;
semilogy(1:numSamplesTest, zTest_Transformed(:,2),'.','MarkerSize',ms+5,'Color',BLUEGREEN); hold on;
leg=legend('Testing Data','NN Prediction (Testing)');
xlabel('Testing Sample ID');
ylabel('Power');
axis([0 numSamplesTest+1 0 1.05*max(TESTING_OUTPUTS(:,2))]);
title('NN Model: Testing');
set(gca,'FontSize',fs);
set(leg,'FontSize',18);
grid on;
grid minor;
%
f4.Position = [100 20 1400 800];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: Scales the TRAINING AND TEST INPUT PARAMETERS from
%                      [min,max] -> [0,1] -> [-1,1]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [TRAINING_INPUTS_SCALED,TESTING_INPUTS_SCALED] = scale_Training_and_Testing_Data(numInputs,TRAINING_INPUTS,TESTING_INPUTS)

%-----------------------------------------------------
%                 Allocate Storage
%-----------------------------------------------------
TRAINING_INPUTS_SCALED = TRAINING_INPUTS;
TESTING_INPUTS_SCALED = TESTING_INPUTS;

%-----------------------------------------------------
%
%     >>>>> Actual 5-D Parameter Space <<<<<
%
%-----------------------------------------------------
% TAIL AMPLITUDE
p1_min = 0.05;
p1_max = 0.30;

% HEAD AMPLITUDE/TAIL AMPLITUDE RATIO
p2_min = 0.0;
p2_max = 0.80;

% UNDULATORY WAVELENGTH
p3_min = 0.5;
p3_max = 1.5;

% UNDULATION FREQUENCY
p4_min = 1.0;
p4_max = 5.0;

% BODY TAPER PT
p5_min = 0.0;
p5_max = 1.0;

% STORE VALUES IN VECTOR FOR AUTOMATED SCALING
SCALE_MIN = [p1_min p2_min p3_min p4_min p5_min];
SCALE_MAX = [p1_max p2_max p3_max p4_max p5_max];

%---------------------------------------------------
%                       FIRST: 
%      Scale INPUT values from [min,max]->[0,1]
%---------------------------------------------------
for i=1:numInputs

    %-----------------------------------------
    % Scale to [0,1]: [1]  min_Z*m + b = 0
    %                 [2]  max_Z*m + b = 1
    %-----------------------------------------
    min_Z = SCALE_MIN(i);
    max_Z = SCALE_MAX(i);
    %
    m1 = 1/(max_Z-min_Z);
    b1 = 1-max_Z*m1;
    %
    TRAINING_INPUTS_SCALED(:,i) = m1 * TRAINING_INPUTS(:,i) + b1;
    TESTING_INPUTS_SCALED(:,i) =  m1 * TESTING_INPUTS(:,i) + b1;

end

%---------------------------------------------------
%                      SECOND: 
%    Scale the INPUT variables from [0,1]->[-1,1]
%---------------------------------------------------
m2=2;
b2=-1;
TRAINING_INPUTS_SCALED(:,1:numInputs) = m2 * TRAINING_INPUTS_SCALED(:,1:numInputs) + b2;
TESTING_INPUTS_SCALED(:,1:numInputs) =  m2 * TESTING_INPUTS_SCALED(:,1:numInputs) + b2;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: computes Error Statistics and Prints Information for NN
%           Validation and Testing
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function print_Error_Information(zData,DATA,strKind,numOutputs)

%------------------------------------------
% STRING ARRAY OF OUTPUT NAMES
%------------------------------------------
strVec = {'Speed','Power'};

%------------------------------------------
% LOOP OVER ALL MODEL OUTPUTS
%------------------------------------------
for n=1:numOutputs

    strMetric = strVec{n};
    
    % FIRST COMPUTE ERRORS
    errABS = abs( zData(:,n) - DATA(:,n) );    % Calculate Absolute Error
    errREL = abs( errABS ./ DATA(:,n) ) * 100; % Calculate Relative Error Percentage
    %
    inds = isinf(errREL); % finds inds if divide by zero occurs
    errREL(inds) = 555;   % if divide by zero happens, revert to 100% error.
    %                     %        NOTE: will happen if output data is 0
    %
    % THEN PRINT INFO TO SCREEN
    %
    stringString = ['*** ' strKind ':' strMetric ' (ERROR INFO) ***\n'];
    %
    fprintf('\n\n***---------------------------***\n');
    fprintf(stringString);
    fprintf('***---------------------------***\n\n');
    %
    fprintf(' --> ABSOLUTE ERROR  <-- \n');
    fprintf('  Max ABS Error: %.4f\n', max( errABS(:,1) ) );
    fprintf('  Min ABS Error: %.4f\n', min( errABS(:,1) ) );
    fprintf('  Avg ABS Error: %.4f\n', mean( errABS(:,1) ) );
    fprintf('  Avg ABS Error: %.4f\n', mean( errABS(:,1) ) );
    fprintf('StDev ABS Error: %.4f\n', std( errABS(:,1) ) );
    fprintf('\n');
    %
    fprintf(' --> RELATIVE PERCENT ERROR  <-- \n');
    fprintf('  Max Percent Error: %.4f\n', max( errREL(:,1) ) );
    fprintf('  Min Percent Error: %.4f\n', min( errREL(:,1) ) );
    fprintf('  Med Percent Error: %.4f\n', median( errREL(:,1) ) );
    fprintf('  Avg Percent Error: %.4f\n', mean( errREL(:,1) ) );
    fprintf('StDev Percent Error: %.4f\n', std( errREL(:,1) ) );
    fprintf('\n');

end