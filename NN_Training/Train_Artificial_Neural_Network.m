%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: TRAINS a neural network w/ 2 hidden layers and biases
%
%     NN Hyperparameters: (+) num_hidden_layer_neurons
%                         (+) learning rate: initial SGD step-size
%                                  NOTE: uses pseudo adaptive learning rate 
%                         (+) lr_adaptive_int: interval for how often to 
%                                  decrease the learning rate
%                         (+) maxIter: # of full epochs for SGD
%                         (+) batch_size: amount of data to use in each mini
%                                  batch of stochastic gradient descent
%                         (+) regularize_Flag: choice of regularization
%                         (+) regularize_Penalty: regularization penalty-term
%
%   Author: Nick A Battista
%   Institution: The College of New Jersey
%   Date: April 2024
%                      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W1_Save,W2_Save,WEnd_Save,b1_Save,b2_Save,costVec,min_Cost] = Train_Artificial_Neural_Network(TRAINING_INPUTS,TRAINING_OUTPUTS,hyper_param_vec)


%------------------------------------------------------
% FLAG: use previous NN weights
%------------------------------------------------------
flag_Use_Stored_Weights = 0;

if flag_Use_Stored_Weights
    fprintf('     - Loading previously stored weights...\n');
    fprintf('     - Loading previously stored biases...\n');
    fprintf('        (Trying to make them more accurate...)\n');
    fprintf('#---------------------------------------------------\n\n');

else
    fprintf('     - NOT loading previously stored weights...\n');
    fprintf('     - NOT loading previously stored biases...\n');
    fprintf('#---------------------------------------------------\n\n');

end

%-----------------------------------------------------------
%                   NN Training Parameters
% hyper_param_vec(1): num_hidden_layer_neurons
% hyper_param_vec(2): learning_rate
% hyper_param_vec(3): lr_adaptive_int
% hyper_param_vec(4): number of epochs
% hyper_param_vec(5): batch_size
% hyper_param_vec(6): print_Interval
% hyper_param_vec(7): # of input parameters
% hyper_param_vec(8): regularization flag {0,1,2}-->{none,L1,L2}
% hyper_param_vec(9): regularization penalty coefficient
%-----------------------------------------------------------

%-----------------------------------------------------------
%                   Redefine NN Training Data
%-----------------------------------------------------------
x0 = TRAINING_INPUTS';   % should be of size: numInputs x numData
z0 = TRAINING_OUTPUTS';  % should be of size: numOutputs x numData ;
%
Ninput = hyper_param_vec(7); % Amount of INPUT neurons
Noutput = length(z0(:,1));   % Amount of OUTPUT neurons
numTrain = length(x0(1,:));  % Amount of Training Data
%
num_hid_layer_neurons = hyper_param_vec(1); % # of neurons in each hidden layer

%-----------------------------------------------------------
%            Get Number of TOTAL OUTPUT ELEMENTS 
%                   (for averaging loss/cost)
%-----------------------------------------------------------
num_Z = numel( z0 );  % # of TOTAL data elements in output matrix


%---------------------------------------------------------------
%---------------------------------------------------------------
% Initialize Weights and Biases
%---------------------------------------------------------------
%---------------------------------------------------------------
if flag_Use_Stored_Weights 
    
    % Will load {W1,W2,WEnd} +  {b1,b2} + {num_hid_layer_neurons}
    load('Trained_Neural_Network_Weights/Trained_Neural_Network.mat');
    clear costVec;
    
else
    
    %--------------------------------------------------------
    %         INITIALIZE WEIGHTS/BIASES RANDOMLY
    %--------------------------------------------------------
    coeff = 1 / sqrt( numTrain ); % for randomizing initial values
    W1 = coeff*( 2*rand( num_hid_layer_neurons, Ninput)-1 );
    W2 = coeff*( 2*rand( num_hid_layer_neurons, num_hid_layer_neurons)-1 );
    WEnd = coeff*( 2*rand( Noutput, num_hid_layer_neurons )-1 );
    b1 = coeff*( 2*rand( num_hid_layer_neurons, 1 )-1 );   
    b2 = coeff*( 2*rand( num_hid_layer_neurons, 1 )-1 );
    
end
%

%-------------------------------------------------------------
%       INITIALIZE GRADIENTS OF WEIGHT/BIAS MATRICES 
%-------------------------------------------------------------
W1_p = zeros( num_hid_layer_neurons, Ninput );
dJ_dW1_p = zeros( num_hid_layer_neurons, Ninput );
%
W2_p = zeros( num_hid_layer_neurons, num_hid_layer_neurons );
dJ_dW2_p = zeros( num_hid_layer_neurons, num_hid_layer_neurons );
%
WEnd_p = zeros( Noutput, num_hid_layer_neurons );
dJ_dWEnd_p = zeros( Noutput, num_hid_layer_neurons );
%
b1_p = zeros( num_hid_layer_neurons, 1 );
dJ_db1_p = zeros( num_hid_layer_neurons, 1 );
%
b2_p = zeros( num_hid_layer_neurons, 1 );
dJ_db2_p = zeros( num_hid_layer_neurons, 1 );


%-------------------------------------------------------------
%           Initialize Learning Rates and Momentum
%-------------------------------------------------------------
% Weight Learning Rates
lambda_1 = hyper_param_vec(2);     % initial learning rate
lambda_2 = hyper_param_vec(2);     % initial learning rate
lambda_End = hyper_param_vec(2);   % initial learning rate
% Bias Learning Rates
lambda_b1 = hyper_param_vec(2);    % initial learning rate
lambda_b2 = hyper_param_vec(2);    % initial learning rate
% Minimum Lambda 
minLAM = hyper_param_vec(2);       % MINIMAL learning rate


%------------------------------------------------
% BATCH SIZE FOR MINIBATCH GRADIENT DESCENT
%           (vein of Stochastic Grad. Descent)
%------------------------------------------------
batch_size_Save = hyper_param_vec(5);


%------------------------------------------------
%        Define Regularization Variables
%------------------------------------------------
regularize_Flag = hyper_param_vec(8); % regularization flag: 
                                      % 0 (none), 1 (L1-LASSO), 2 (L2-RIDGE) 
lam_regularize = hyper_param_vec(9);  % regularization penalty coefficient

                          
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%               >>>> START NEURAL NETWORK TRAINING! <<<<
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%
numEpochs = hyper_param_vec(4);   % max # of EPOCHS to TRAIN model
costVec = zeros( 1, numEpochs );  % initalize cost function storage
%
for epochIter=1:numEpochs
    
  
    
    %-----------------------------------------------------------
    %               PSEUDO-ADAPTIVE LEARNING RATE
    %         ("decrease learning rate every X epochs")
    %-----------------------------------------------------------
    if mod( epochIter, hyper_param_vec(3) )==0
        minLAM = 0.5*minLAM;
        pause(4);
    elseif mod(epochIter,50)==0
        pause(4);
    end
    %
    lambda_1 = minLAM;
    lambda_2 = minLAM;
    lambda_End = minLAM;
    lambda_b1 = minLAM;
    lambda_b2 = minLAM;
    
    
    %-----------------------------------------------------------
    % RANDOMLY SHUFFLE INDICES for MINI-BATCH RANDOM SAMPLING
    %       and RESET EPOCH PARAMETERS
    %-----------------------------------------------------------
    indsRandom = randperm(length(1:numTrain));  % Randomly shuffle training data indices for SGD
    costSum = 0;                                % Reset SINGLE Epoch cost to 0
    batch_size = batch_size_Save;               % Reset to original batch size

    
    %------------------------------------------------------
    % Iteration Number Inside SINGLE Epoch
    %------------------------------------------------------
    for iter=1:floor(numTrain/batch_size_Save)


        %-----------------------------------------------------------
        % RANDOMLY SHUFFLE INDICES for MINI-BATCH RANDOM SAMPLING
        %-----------------------------------------------------------
        if iter ~= floor(numTrain/batch_size)
            inds = indsRandom( 1+(iter-1)*batch_size:batch_size*iter);
        else
            inds = indsRandom( 1+(iter-1)*batch_size:end);
            batch_size = length(inds);
        end


        %-----------------------------------------------------
        %                  Forward Propagation
        %-----------------------------------------------------
        [zHat,x2,x1] = forward_propagate(x0(:,inds),W1,W2,WEnd,b1,b2);


        %-----------------------------------------------------
        %   Compute Cost Function (w/ or w/o regularization)
        %-----------------------------------------------------
        %
        % Vectorized Mean-Squared Error (MSE) Cost/Loss 
        %
        J = 0.5*( z0(:,inds) - zHat ).^2;  
        %
        % Regularize the Cost Function ----------------------
        %
        if strcmp(regularize_Flag,'L2')
            cost =  ( sum(sum(J) ) + 0.5*lam_regularize*( sum( sum( W1.^2 ) )  + sum( sum( WEnd.^2 ) ) + sum( sum( b1.^2 ) )    ) );
        elseif strcmp(regularize_Flag,'L1') 
            cost = ( sum(sum(J) ) + 0.5*lam_regularize*( sum( sum( abs(W1) ) ) + sum( sum( abs( WEnd ) ) ) + sum( sum( abs(b1) ) )  ) ); 
        else
            cost = ( sum( sum(J) ) ); 
        end    
        %
        costSum = costSum + cost; % cumulative cost across epoch


        
        %-----------------------------------------------------
        %               COMPUTE DELTA MATRICES
        %       --> uses f_out = x; so f'_{out}=1
        %-----------------------------------------------------
        delta_END = -( z0(:,inds) - zHat );
        delta_2 = (WEnd'*delta_END) .* act_function_PRIME( W2*x1 + b2 );
        delta_1 =     (W2'*delta_2) .* act_function_PRIME( W1*x0(:,inds) + b1 );
        
        
        %-----------------------------------------------------
        %           Compute REGULARIZATION Gradients
        %-----------------------------------------------------
        if strcmp( regularize_Flag , 'L2' )
            regularize_grad_W1 = lam_regularize*( W1 );
            regularize_grad_W2 = lam_regularize*( W2 );
            regularize_grad_WEnd = lam_regularize*( WEnd );
            regularize_grad_b1 = lam_regularize*( b1 );
            regularize_grad_b2 = lam_regularize*( b2 );
        elseif strcmp( regularize_Flag , 'L1' )
            regularize_grad_W1 = lam_regularize*( sign(W1) );
            regularize_grad_W2 = lam_regularize*( sign(W2) );
            regularize_grad_WEnd = lam_regularize*( sign(WEnd) );
            regularize_grad_b1 = lam_regularize*( sign(b1) );
            regularize_grad_b2 = lam_regularize*( sign(b2) );
        else
            regularize_grad_W1 = 0;
            regularize_grad_W2 = 0;
            regularize_grad_WEnd = 0;
            regularize_grad_b1 = 0;
            regularize_grad_b2 = 0;
        end


        %-------------------------------------------------------------
        %            COMPUTE GRADIENTS FOR BACK PROPAGATION!  
        %-------------------------------------------------------------
        %
        dJ_dWEnd = 1/batch_size * ( ( delta_END * x2' ) + regularize_grad_WEnd  );           
        %
        dJ_dW2 =   1/batch_size * ( delta_2 * x1'  + regularize_grad_W2  );
        %
        dJ_dW1 =   1/batch_size * ( delta_1 * x0(:,inds)'  + regularize_grad_W1  );
        %
        dJ_db2 =   1/batch_size * ( delta_2 * ones(batch_size,1)  + regularize_grad_b2 );
        %
        dJ_db1 =   1/batch_size * ( delta_1 * ones(batch_size,1)  + regularize_grad_b1 );

 

        %-----------------------------------------------------
        %               Perform Gradient Descent 
        %-----------------------------------------------------
        %
        % BACK PROP. FOR WEIGHTS!
        %
        W1 = W1 - lambda_1 * ( dJ_dW1 ); 
        %
        W2 = W2 - lambda_2 * ( dJ_dW2 ); 
        %
        WEnd = WEnd - lambda_End * ( dJ_dWEnd ); 
        %
        % BACK PROP. FOR BIASES! 
        %
        b1 = b1 - lambda_b1 * ( dJ_db1 ); 
        %
        b2 = b2 - lambda_b2 * ( dJ_db2 ); 

        
    end
    
    %-------------------------------------------------
    %          Save COST for a SINGLE Epoch
    %-------------------------------------------------
    costVec(epochIter) = costSum / num_Z; % Store Cost For Full SINGLE EPOCH

    
    %-------------------------------------------------
    %       Print Training Information to Screen
    %-------------------------------------------------
    if mod( epochIter , hyper_param_vec(6) )==0
        fprintf('#--------------------------------------\n');
        fprintf('    *** Epoch: %d *** \n',epochIter);
        fprintf('Cost (J) = %.8f\n\n',costVec(epochIter) );
    end
    
end

fprintf('\n#---------------------------------------------------');
fprintf('         >>>> NN MODEL TRAINING DONE <<<< \n');
fprintf('#---------------------------------------------------');

%-----------------------------
% Data to Give Back to User
%-----------------------------
W1_Save = W1;
W2_Save = W2;
WEnd_Save = WEnd;
%
b1_Save = b1;
b2_Save = b2;
%
min_Cost = cost;


