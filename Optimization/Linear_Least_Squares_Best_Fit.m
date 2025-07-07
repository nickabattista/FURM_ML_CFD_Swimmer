%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: use Linear Least Squares to find the best fit cubic 
%           polynomial through data {x_j,y_j}
%
% NOTE: 
%        (1) Fit data to: y = f(x;c) = c0 + c1*x + c2*x^2 + c3^x^3
%        (2) Compute DATA matrix for Linear Least Squares
%        (3) Use pseudo-inverse approach to find coefficients
%        (4) Will plot data to check :)
%        (5) Compute the errors (L1, L2, etc.)
%
%   Author: Nick A. Battista
%   Date: November 2021
%   Institution: The College of New Jersey
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function Linear_Least_Squares_Best_Fit()


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
%                 Setup linear problem matrix: 
%                 
%                  MAT*coeffsVec = XY_Data(:,2) 
%------------------------------------------------------------
MAT = zeros( NSamples, 4);       % allocate storage to MAT
for j=1:NSamples                 % loop over all data samples
    x = XY_Data(j,1);            % x-value from jth sample point
    MAT(j,:) = [1  x  x^2  x^3]; % fill row of matrix 
end


%------------------------------------------------------------
%           SOLVE THE LINEAR LEAST SQUARES PROBLEM: 
%             >>> Compute the pseudo-inverse <<<
%------------------------------------------------------------
coeffsVec = inv(MAT'*MAT) * MAT' * XY_Data(:,2);

%------------------------------------------------------------
%       Print out best fit coefficients to the screen
%------------------------------------------------------------
fprintf('\n---------------------------------------------------\n\n');
fprintf('BEST FIT COEFFICIENTS:\n');
for i=1:length(coeffsVec)
   strTMP = ['     c_' num2str(i-1) ' = ' num2str(coeffsVec(i)) '\n'];
   fprintf(strTMP);
end


%------------------------------------------------
%               Plot Attributes
%------------------------------------------------
lw = 6;  % LineWidth
ms = 30; % MarkerSize
fs = 23; % FontSize

%------------------------------------------------------------
%               FIGURE 1: BEST-FIT vs Raw Data
%------------------------------------------------------------
%
% Get independent variable array to input into model
xVec  = linspace( min(XY_Data(:,1)), max( XY_Data(:,1)), 100); 
%
% Predict y-Values on xVec using best fit coefficients
yPredVec = coeffsVec(1) + coeffsVec(2)*xVec + coeffsVec(3)*xVec.^2 + coeffsVec(4)*xVec.^3;
%
f1=figure(1);
%
plot(XY_Data(:,1),XY_Data(:,2),'.','MarkerSize',ms); hold on;
plot(xVec,yPredVec,'-','LineWidth',lw); hold on;
legend('\{(x_j,y_j)\} data','LS Fit','Location','NorthEast');
xlabel('x');
ylabel('y');
set(gca,'FontSize',fs);
grid on;
grid minor;

%------------------------------------------------
%           Position figures on screen
%------------------------------------------------
f1.Position = [10  200 600 450];


%-------------------------------------------------------------------
%                           ERROR ANALYSIS: 
%             (Residuals, L1-Norm, L2-Norm, LInf-Norm etc)
%-------------------------------------------------------------------
r = XY_Data(:,2) - MAT*coeffsVec; % Actual residual from Least Squares Soln
L1_Norm = sum( abs(r) );          % Calculate L1-Norm
L2_Norm = sqrt( r'*r );           % Calculate L2-Norm
LInf_Norm = max( abs(r) );        % Calculate LInfinity-Norm
%
fprintf('\n---------------------------------------------------\n\n');
fprintf('LEAST SQUARE SOLUTION ERROR ANALYSIS:\n\n');
fprintf('   -> The L1-Norm of the Residual is: %.3f\n',L1_Norm);
fprintf('   -> The L2-Norm of the Residual is: %.3f\n',L2_Norm);
fprintf('   -> The LINF-Norm of the Residual is: %.3f\n',LInf_Norm);
fprintf('\n---------------------------------------------------\n\n');


