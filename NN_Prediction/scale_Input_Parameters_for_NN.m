
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: Scales the MODEL INPUT PARAMETERS from [min,max] -> [0,1] -> [-1,1]
%
%   Author: Nick A Battista
%   Institution: The College of New Jersey
%   Date: April 2024
%       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function INPUT_PARAMS_SCALED = scale_Input_Parameters_for_NN(INPUT_PARAMS)

%-----------------------------------------------------
%                 Allocate Storage
%-----------------------------------------------------
INPUT_PARAMS_SCALED = INPUT_PARAMS;


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
for i=1:length(INPUT_PARAMS(1,:))

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
    INPUT_PARAMS_SCALED(:,i) = m1 * INPUT_PARAMS(:,i) + b1;

end


%---------------------------------------------------
%                      SECOND: 
%    Scale the INPUT variables from [0,1]->[-1,1]
%---------------------------------------------------
m2=2;
b2=-1;
INPUT_PARAMS_SCALED = m2 * INPUT_PARAMS_SCALED + b2;

