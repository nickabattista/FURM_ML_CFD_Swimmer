%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: performs STOCHASTIC GRADIENT DESCENT to find the best fit  
%           cubic polynomial through data {x_j,y_j}
%
%       Model Function: y(x) = c0 + c1*x + c2*x^2 + c3*x^3
%
%       Cost/Loss Function (MSE): 0.5 * SUM_j ( y_j - z(x) )^2 
%
%       Note: for this example, the best fit coefficients are: 
%              c_0,c1,c2,c3 = {-1.6648,1.1035,2.1508,-1.097}
%
%   Author: Nick A. Battista
%   Date: November 2021
%   Institution: The College of New Jersey
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Stochastic_Gradient_Descent_Best_Fit()

%------------------------------------------------------------
%        Initial Guess for Coefficients: c0, c1, c2, c3
%------------------------------------------------------------
coeffsInitialGuess = 0.1*[1 1 1 1];

%------------------------------------------------------------
%               # of EPOCHS to iterate over
%------------------------------------------------------------
NEpochs = 2500;

%------------------------------------------------------------
%               Learning rate (step-size)
%------------------------------------------------------------
alpha = 1e-4;


%------------------------------------------------------------
%                LOAD DATA: XY_Data <--> {(x,y)} 
%   
%       XY_Data: col 1: x 
%                col 2: y
%                  row: different data point
%------------------------------------------------------------
load('Best_Fit_Ex1_Data.mat');
%
NSamples = length(XY_Data(:,1)); % # of data samples


%------------------------------------------------------------
%  Define current estimate (err 'previous') as initial guess
%------------------------------------------------------------
coeffsPrev = coeffsInitialGuess;


%------------------------------------------------------------
%          Initialize Storage Vectors and Arrays
%------------------------------------------------------------
coeffsMAT = zeros(NEpochs,4); 
lossVec = zeros(NEpochs,1);


%------------------------------------------------
%           Calculate COST/LOSS
%------------------------------------------------
loss=0;
for j=1:NSamples
    x = XY_Data(j,1);                        % jth x-point
    y = XY_Data(j,2);                        % jth y-point
    yHat = MODEL( coeffsPrev, x );           % evaluate current model
    loss = loss + ( XY_Data(j,2) - yHat )^2; % loss for jth data point
end
loss = 0.5*loss;

%------------------------------------------------------------
%           For storing each iteration's guess
%------------------------------------------------------------
ct = 1;
coeffsMAT(ct,:) = coeffsPrev; 
lossVec(ct,1) = loss;


%--------------------------------------------------------------------
%--------------------------------------------------------------------
%            Perform STOCHASTIC Grad Descent --> iterate!
%--------------------------------------------------------------------
%--------------------------------------------------------------------
for n=1:NEpochs

    
    %-----------------------------------------------------------
    %       RANDOMLY SHUFFLE SAMPLE INDICES AND DATA
    %-----------------------------------------------------------
    IDs = randperm(NSamples);     % Randomly shuffle training data IDs for SGD
    XY_Shuffled = XY_Data(IDs,:); % Randomly shuffle training data

    
    %-----------------------------------------------------------
    %             Iterate Across a SINGLE Epoch
    %-----------------------------------------------------------
    for i=1:NSamples    
    
        %---------------------------------------------------------
        %               Get next coefficient guess 
        % (SGD step for single selected training sample in epoch)
        %---------------------------------------------------------
        c0_Next = coeffsPrev(1) - alpha * GRAD( coeffsPrev, XY_Shuffled(i,:), 0); % "Partial loss, partial c0"
        c1_Next = coeffsPrev(2) - alpha * GRAD( coeffsPrev, XY_Shuffled(i,:), 1); % "Partial loss, partial c1"
        c2_Next = coeffsPrev(3) - alpha * GRAD( coeffsPrev, XY_Shuffled(i,:), 2); % "Partial loss, partial c2"
        c3_Next = coeffsPrev(4) - alpha * GRAD( coeffsPrev, XY_Shuffled(i,:), 3); % "Partial loss, partial c3"
        %
        coeffsNext = [c0_Next c1_Next c2_Next c3_Next];

        %------------------------------------------------
        %               Change 'who is who'
        %  (redefine prev. as current for next iteration)
        %------------------------------------------------
        coeffsPrev = coeffsNext;

    end
    
    %---------------------------------------------------------------
    %             Calculate COST/LOSS once epoch is over
    %---------------------------------------------------------------
    loss=0;
    for j=1:NSamples
        x = XY_Shuffled(j,1);           % x-value from jth sample point
        y = XY_Shuffled(j,2);           % y-value from jth sample point        
        yHat = MODEL( coeffsPrev, x );  % evaluate current model
        loss = loss + ( y - yHat )^2;   % loss for jth data point
    end
    
    %---------------------------------------------------------------
    %      Store next guess into matrix once epoch is over
    %---------------------------------------------------------------
    ct=ct+1;
    coeffsMAT(ct,:) = coeffsPrev;
    lossVec(ct) = 0.5*loss;    

end

%------------------------------------------------
%       Print out each epoch's information
%------------------------------------------------
fprintf('\n---------------------------------------------------\n\n');
fprintf('SGD INFORMATION (epoch, loss, coefficients):\n\n');
fprintf('n | loss | (c_0^n,c_1^n,c_2^n,c_3^n) \n')
for n=1:200:length(coeffsMAT(:,1))
    fprintf('%d | %.4f |(%.4f,%.4f,%.4f,%.4f)\n', n-1, lossVec(n), coeffsMAT(n,1), coeffsMAT(n,2), coeffsMAT(n,3), coeffsMAT(n,4) );
end



%------------------------------------------------
%               Plot Attributes
%------------------------------------------------
lw = 6;  % LineWidth
ms = 30; % MarkerSize
fs = 23; % FontSize


%------------------------------------------------
%   FIGURE 1: Cost/Loss vs Iteration Number
%------------------------------------------------
f1=figure(1);
loglog(0:NEpochs,lossVec,'.-','LineWidth',lw,'MarkerSize',ms); hold on;
xlabel('Iteration Number');
ylabel('Cost/Loss');
set(gca,'FontSize',fs);
grid on; grid minor;


%------------------------------------------------
%   FIGURE 2: BEST-FIT vs Raw Data
%------------------------------------------------
% Get independent variable array to input into model
xVec  = linspace( min(XY_Data(:,1)), max( XY_Data(:,1)), 100); 
%
% Predict y-Values on xVec using last coefficients
yPredVec = MODEL( coeffsNext, xVec );
%
f2=figure(2);
plot(XY_Data(:,1),XY_Data(:,2),'.','MarkerSize',ms); hold on;
plot(xVec,yPredVec,'-','LineWidth',lw); hold on;
xlabel('x');
ylabel('y');
legend('Raw Data','Best Fit');
set(gca,'FontSize',fs);
grid on;
grid minor;


%------------------------------------------------
%          Position figures on screen
%------------------------------------------------
f1.Position = [10  500 900 450];
f2.Position = [900 500 600 450];



%-------------------------------------------------------------------
%                           ERROR ANALYSIS: 
%             (Residuals, L1-Norm, L2-Norm, LInf-Norm etc)
%-------------------------------------------------------------------
r = XY_Data(:,2) - MODEL( coeffsNext, XY_Data(:,1) ); % Residuals from model fit 
L1_Norm = sum( abs(r) );       % Calculate L1-Norm
L2_Norm = sqrt( r'*r );        % Calculate L2-Norm
LInf_Norm = max( abs(r) );     % Calculate LInfinity-Norm
%
fprintf('\n---------------------------------------------------\n\n');
fprintf('LEAST SQUARE SOLUTION ERROR ANALYSIS:\n\n');
fprintf('   -> The L1-Norm of the Residual is: %.3f\n',L1_Norm);
fprintf('   -> The L2-Norm of the Residual is: %.3f\n',L2_Norm);
fprintf('   -> The LINF-Norm of the Residual is: %.3f\n',LInf_Norm);
fprintf('\n---------------------------------------------------\n\n');





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: calculates Model Function for coefficients \vec{c} at a   
%           specific sample point (x,y), e.g., 
%
%           Model: yHat = c0 + c1*x + c2*x^2 + c3*x^3
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function yHat = MODEL( coeffs, x )

    yHat = coeffs(1) + coeffs(2)*x + coeffs(3)*x.^2 + coeffs(4)*x.^3;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: provide gradient of the loss/cost function being minimized 
%   
%       --> Recall the gradient is respect to unknown coefficients 
%           c0, c1, c2, and c3.  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function gradVal = GRAD( coeffsVec, XY_Data, whichPartial )

%---------------------------------------------------
% Initialize sum variable
%---------------------------------------------------
gradVal = 0;

%---------------------------------------------------
% Partial Derivative w/ respect to coefficient c0
%---------------------------------------------------
if whichPartial == 0
    
    for j=1:length( XY_Data(:,1) )
        x = XY_Data(j,1);            % x-value from jth sample point        
        y = XY_Data(j,2);            % y-value from jth sample
        yHat = MODEL( coeffsVec, x );        
        gradVal = gradVal - (y - yHat);      
    end

%---------------------------------------------------
% Partial Derivative w/ respect to coefficient c1
%---------------------------------------------------    
elseif whichPartial == 1
    
    for j=1:length( XY_Data(:,1) )
        x = XY_Data(j,1);            % x-value from jth sample point        
        y = XY_Data(j,2);            % y-value from jth sample
        yHat = MODEL( coeffsVec, x );        
        gradVal = gradVal - (y - yHat) * x^1;      
    end    

%---------------------------------------------------
% Partial Derivative w/ respect to coefficient c2
%---------------------------------------------------    
elseif whichPartial == 2
    
    for j=1:length( XY_Data(:,1) )
        x = XY_Data(j,1);            % x-value from jth sample point        
        y = XY_Data(j,2);            % y-value from jth sample
        yHat = MODEL( coeffsVec, x );        
        gradVal = gradVal - (y - yHat) * x^2;      
    end     

%---------------------------------------------------
% Partial Derivative w/ respect to coefficient c3
%---------------------------------------------------    
elseif whichPartial == 3
    
    for j=1:length( XY_Data(:,1) )
        x = XY_Data(j,1);            % x-value from jth sample point        
        y = XY_Data(j,2);            % y-value from jth sample
        yHat = MODEL( coeffsVec, x );        
        gradVal = gradVal - (y - yHat) * x^3;      
    end     
    
end
