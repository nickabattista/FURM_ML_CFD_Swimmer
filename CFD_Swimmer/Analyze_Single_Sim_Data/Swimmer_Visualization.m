%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: visualizes the swimmer horizontally moving across the domain
%           (default visualizes the swimmer geometry and fluid vorticity)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Swimmer_Visualization()

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
dx = Lx/Nx;           % grid (spatial) step-size in x-direction
dy = Ly/Ny;           % grid (spatial) step-size in y-direction
ds = 0.5*(Lx/Nx);     % Lagrangian spacing
%
xVec = 0:dx:Lx-dx;         % Vector of grid cells in x-direction
yVec = 0:dy:Ly-dy;         % Vector of grid cells in y-direction
[X,Y]=meshgrid(xVec,yVec); % Meshing variables for visualization;


%----------------------------------------------
%             Plotting Attributes
%----------------------------------------------
ms = 55; % MarkerSize
fs = 23; % FontSize

%----------------------------------------------
%      Define colormap for fluid data
%----------------------------------------------
cMAP=colormap('pink');                % built-in MATLAB colormap
cMAP=[cMAP(:,3) cMAP(:,2) cMAP(:,1)]; % modify it from "pink"->"light blue"
colormap(cMAP);                       % redefine selected colormap


%------------------------------------------------------------------------
%------------------------------------------------------------------------
%                Loop and visualize selected time-points
%------------------------------------------------------------------------
%------------------------------------------------------------------------
for i=start:1:finish-1

        %-------------------------------------------
        % Print information to screen
        %-------------------------------------------
        if mod(i,10)==0
            fprintf('Analyzing i = %d of %d\n',i,finish);
        end
        
        %-------------------------------------------------------
        % Create string timepoint for vtk data being analyzed
        %-------------------------------------------------------
        if i<10
           strTimePoint = ['000', num2str(i) ];
        elseif i<100
           strTimePoint = ['00', num2str(i) ];
        elseif i<1000
           strTimePoint = ['0', num2str(i)];
        else
           strTimePoint = num2str(i);
        end        

        
        %----------------------------------------------------
        % LOAD LAGRANGIAN DATA: {lagPts,uLag,vLag,F_Lag}
        %----------------------------------------------------
        load(['LAG_INFO.' num2str(i) '.mat']); 
        XY_C = lagPts;
        clear uLag vLag F_Lag lagPts;
         
        
        %-------------------------------------------------------------
        % LOAD EULERIAN DATA: default is setup for Omega (vorticity)
        %-------------------------------------------------------------
        EulerianFileName = [path_To 'Omega.' strTimePoint '.vtk'];
        Vorticity = read_Eulerian_Data_From_vtk( EulerianFileName );

        
        %--------------------------------------------------------------
        %--------------------------------------------------------------
        %                     VISUALIZE THE DATA!
        %--------------------------------------------------------------
        %--------------------------------------------------------------
        f1 = figure(1);
        
        
        %--------------------------------------------------------------
        %          Plot Fluid Data (default is vorticity)
        %        (note: vorticity = curl of velocity field)
        %--------------------------------------------------------------
        %
        surf(X, Y, Vorticity, 'edgecolor','none'); hold on;
        %
        minThreshold = -25;
        maxThreshold = 25;
        %
        h=colorbar;
        caxis([minThreshold maxThreshold])
        h.Ticks = linspace( minThreshold, maxThreshold, 5 );

        %--------------------------------------------------------------
        %                   Plot Swimmer Geometry
        %
        %   (note: to plot geometry above fluid data, need to 
        %   "place on it top" of fluid data, zVec accomplishes this)
        %--------------------------------------------------------------
        %
        zVec = 1.05*max(max(Vorticity))*ones(size(XY_C(:,2))); % to plot lag. data on top
        %
        plot3(XY_C(:,1),XY_C(:,2),zVec,'.','MarkerSize',ms,'Color',[0 0 0]); hold on;
        plot3(XY_C(:,1),XY_C(:,2),zVec,'.','MarkerSize',ms-23,'Color',[1 1 1]); hold on;

        %--------------------------------------------------------------
        % Plot labels, plots axes, sizing/location on screen
        %--------------------------------------------------------------
        set(gca,'FontSize',fs);
        xlabel('x');
        ylabel('y');
        axis([0 Lx 0 Ly]);
        %
        f1.Position = [50 10 1200 385];        
        view(2);
        
        %--------------------------------------------------------------
        % Pause on frame (fraction of second) and clear prev fig
        %--------------------------------------------------------------
        pause(0.1);
        hold off;
       
        
end % Ends for-loop over all time-points being visualized


%------------------------------------------------------
%           REMOVE PATH TO SIMULATION DATA
%------------------------------------------------------
rmpath(path_To);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: Reads in EULERIAN scalar data from .vtk format            
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function e_Data = read_Eulerian_Data_From_vtk(filename)

%------------------------------------------------------
% filename --> EULERIAN-DATA.xxxx.vtk 
%------------------------------------------------------
fileID = fopen(filename);
if ( fileID== -1 )
    error('\nCANNOT OPEN THE FILE!');
end

str = fgets(fileID); %-1 if eof
if ~strcmp( str(3:5),'vtk')
    error('\nNot in proper VTK format');
end

% read in the VTK header info %
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);

% Check whether VTK file has time info. This is a VTK file with time, 
%       need to read the next 3 lines to have read in appropriate
%       number of header lines.
if ~strcmp( str(1:3), 'DIM')
	str = fgets(fileID);
    str = fgets(fileID);
    str = fgets(fileID);
end

% Store grid info
N = sscanf(str,'%*s %f %f %*s',2);
Nx = N(1); Ny = N(2);

% bypass lines in header %
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);

% bypass lines in header %
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);

% get formatting for reading in data from .vtk in fscanf %
strVec = '%f';
for i=2:Nx
    strVec = [strVec ' %f'];
end

% read in the vertices %
[e_Data,count] = fscanf(fileID,strVec,Nx*Ny);
if count ~= Nx*Ny
   error('\nProblem reading in Eulerian Data.'); 
end

% reshape the matrix into desired data type %
e_Data = reshape(e_Data, Nx, count/Nx); % Reshape vector -> matrix (every 3 entries in vector make into matrix row)
e_Data = e_Data';                       % Store vertices in new matrix

fclose(fileID);                         % Closes the data file.

clear filename fileID str strVec count analysis_path;




