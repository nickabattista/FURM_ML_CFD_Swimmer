%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: analyzes the swimmer data to look at various quantities of
%           interest, such as speed vs. time, power vs. time, etc.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Swimmer_Analysis()

%------------------------------------------------------
%             WHAT TIME-POINTS TO ANALYZE
%------------------------------------------------------
start=0;          % 1ST interval # included in data analysis
finish=200;       % LAST interval # included in data analysis 


%------------------------------------------------------
%             ADD PATH TO SIMULATION DATA
%------------------------------------------------------
path_To = '../Single_Swimmer_Sim/viz_IB2d/';
addpath(path_To);


%-----------------------------------------------------------------
%            SIMULATION GEOMETRY & TIME INFORMATION 
% LOADS: {BodyLength,numBody,dsBody,dtInput,dpInput,areaEnvelope,...
%         num_Kinematics_Store,TailAmpInput,HeadAmpInput,...
%         HampTampRatioInput,WavelengthInput,FrequencyInput,...
%         EnvelopeBodyPtInput,NxInput,LxInput,NyInput,LyInput}
%-----------------------------------------------------------------
load('../Single_Swimmer_Sim/GEOMETRY_TIME_SIM_INFO.mat');


%------------------------------------------------------
% TEMPORAL & GRID INFO FROM GEOMETRY_TIME_SIM_INFO.mat
%------------------------------------------------------
dt = dtInput;                  % Time-step
dp = dpInput;                  % Print dump
dpdt = dt*dp;                  % Time between stored time-points
%
timeVec=(start:finish-1)*dpdt*FrequencyInput; % Vector containing stroke times
%
Lx = LxInput;         % Length of grid in x-direction
Ly = LyInput;         % Length of grid in y-direction
Nx = NxInput;         % # of grid cells in x-direction
Ny = NyInput;         % # of grid cells in y-direction
ds = 0.5*(Lx/Nx);     % Lagrangian spacing



%------------------------------------------------------
% Allocate Storage for Data Storage Arrays
%------------------------------------------------------
BL_Vec   = zeros(finish-start,1);
xRight_R = zeros(finish-start,1);
xRight_L = zeros(finish-start,1);
xSpeed_R   = zeros(finish-start,1);
xSpeed_L   = zeros(finish-start,1);
powerDIM = zeros(finish-start,1);


%------------------------------------------------------------------------
%------------------------------------------------------------------------
%                  Loop and analyze selected time-points
%------------------------------------------------------------------------
%------------------------------------------------------------------------
for i=start:1:finish-1

        %-------------------------------------------
        % Print information to screen
        %-------------------------------------------
        if mod(i,10)==0
            fprintf('Analyzing i = %d of %d\n',i,finish);
        end

        %-------------------------------------------
        % Define index variable for data storage 
        %   (since MATLAB arrays start at 1, not 0)
        %-------------------------------------------
        ii = i+1;
        
        %--------------------------------------------------------------
        %
        % Compute distance, bodyspeed, power from previous time-step
        %
        %--------------------------------------------------------------
        if i==0

            %---------------------------------------------
            % at t=0, distance traveled and speed are 0
            %---------------------------------------------
            xSpeed_R(ii,1) = 0;
            xSpeed_L(ii,1) = 0;
            xRight_R(ii,1) = 0;
            xRight_L(ii,1) = 0;
            powerDIM(ii,1) = 0;
        
        else
            
            %-------------------------------------------
            % LOAD DATA (for current time-point)
            %        {lagPts,uLag,vLag,F_Lag}
            %------------------------------------------- 
            load(['LAG_INFO.' num2str(i) '.mat']); 
            XY_C = lagPts;
            fX = F_Lag(:,1);
            fY = F_Lag(:,2);
            uBody = uLag;
            vBody = vLag;
            
            %-------------------------------------------
            % LOAD DATA (from previous time-point)
            %        {lagPts,uLag,vLag,F_Lag}
            %------------------------------------------- 
            load(['LAG_INFO.' num2str(i-1) '.mat']); 
            XY_P = lagPts;
            clear uLag vLag F_Lag;

            %--------------------------------------------------------------
            %                       MIDLINE LENGTH
            %--------------------------------------------------------------
            BL_Vec(ii,1) = sum( sqrt( ( XY_C(2:end,1) - XY_C(1:end-1,1) ).^2 + ( XY_C(2:end,2) - XY_C(1:end-1,2) ).^2  ) );

            %-----------------------------------------------------------
            %    Computing horizontal distance between lag positions
            %     (used in distance, speed, and power calculations)
            %-----------------------------------------------------------
            xDiff = XY_C(:,1) -XY_P(:,1);

            %----------------------------------------------------------
            %     Fix for going through domain! (periodic boundary)
            %----------------------------------------------------------
            if ( abs( xDiff(1) ) > 0.5 ) && ( xDiff(1) < 0 )
                xDiff(1) = XY_C(1,1) + Lx - XY_P(1,1);
            elseif ( abs( xDiff(1) ) > 0.5 ) && ( xDiff(1) > 0 )
                xDiff(1) = XY_C(1,1) - Lx - XY_P(1,1);
            end
            %
            if ( abs( xDiff(end) ) > 0.5 ) && ( xDiff(end) < 0 )
                xDiff(end) = XY_C(end,1) + Lx - XY_P(end,1);
            elseif ( abs( xDiff(end) ) > 0.5 ) && ( xDiff(end) > 0 )
                xDiff(end) = XY_C(end,1) - Lx - XY_P(end,1);
            end

            %----------------------------------------------------------
            %           Compute FORWARD SWIMMING DISTANCE
            %----------------------------------------------------------
            xRight_R(ii,1) = xRight_R(ii-1,1) + xDiff(end);
            xRight_L(ii,1) = xRight_L(ii-1,1) + xDiff(1);

            %----------------------------------------------------------
            %           Compute FORWARD SPEED (horizontal)
            %----------------------------------------------------------
            xSpeed_R(ii,1) = xDiff(end) / (dpdt); % measured from head
            xSpeed_L(ii,1) = xDiff(1)   / (dpdt); % measured from tail

            %----------------------------------------------------------
            %           Calculate Power (energy expenditure)
            %           --> P(t) = \int fVec \dot uVec ds
            %----------------------------------------------------------
            powerDIM(ii,1) = ( fX'*uBody + fY'*vBody )*ds;

        end

end % Ends for-loop over all time-points being analyzed


%------------------------------------------------------
%           REMOVE PATH TO SIMULATION DATA
%------------------------------------------------------
rmpath(path_To);


%----------------------------------------------------------
% Translate to set starting position for swimming at zero
%----------------------------------------------------------
xRight_R = xRight_R - xRight_R(1,1);
xRight_L = xRight_L - xRight_L(1,1);

%------------------------------------------------------
%                  PLOT ATTRIBUTES
%------------------------------------------------------
ms=30; % MarkerSize
lw=5;  % LineWidth
fs=23; % FontSize

%------------------------------------------------------
%              FIGURE: Speed vs Time
%------------------------------------------------------
f1=figure(1);
plot(timeVec,xSpeed_R,'-','LineWidth',lw); hold on;
xlabel('Stroke Cycles');
ylabel('Speed');
set(gca,'FontSize',fs);
grid on;
grid minor;

%------------------------------------------------------
%              FIGURE: Power vs Time
%------------------------------------------------------
f2=figure(2);
plot(timeVec,powerDIM,'-','LineWidth',lw); hold on;
xlabel('Stroke Cycles');
ylabel('Power (Energy Expenditure)');
set(gca,'FontSize',fs);
grid on;
grid minor;

%------------------------------------------------------
%         FIGURE: Cost of Transport vs Time
%------------------------------------------------------
f3=figure(3);
plot(timeVec,powerDIM./xSpeed_R,'-','LineWidth',lw); hold on;
xlabel('Stroke Cycles');
ylabel('Cost of Transport');
set(gca,'FontSize',fs);
grid on;
grid minor;

%------------------------------------------------------
%         FIGURE: Midline Length vs Time
%------------------------------------------------------
f4=figure(4);
plot(timeVec,BL_Vec,'-','LineWidth',lw); hold on;
xlabel('Stroke Cycles');
ylabel('Midline Length');
set(gca,'FontSize',fs);
grid on;
grid minor;

%------------------------------------------------------
%             Positions figures on screen
%------------------------------------------------------
f1.Position = [10   600 550 400];
f2.Position = [550  600 550 400];
f3.Position = [10   100 550 400];
f4.Position = [550  100 550 400];








