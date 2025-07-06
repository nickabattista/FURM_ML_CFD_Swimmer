---------------------------------------------------------------------------                               
                                README.txt
---------------------------------------------------------------------------

HOW TO RUN: 

    - run the main2d.m script to run the simulation

PRODUCES: 

    - Folder: viz_IB2d/
            --> Contains data for different time-points in simulation

            --> Lagrangian & Eulerian Data stored to .vtk format 
                for visualization with open-source software, such as
                VisIt (https://visit-dav.github.io/visit-website/index.html).

            --> Lagrangian Data (positions, forces, etc.) stored to .MAT files 
                for analysis in MATLAB

            --> NOTE: as is, only will save Lagrangian data and vorticity data
                 to .vtk format. Can select other data to save in the input2d file

SIMULATION ATTRITUBES:
            
    - Swimmer Case: Tamp = 0.10   (tail amplitude)
                    Hamp = 0.02   (head amplitude)
                    lambda = 1.0  (undulatory wavelength)
                    frequency = 5 (undulatory frequency)
                    xB* = 1.0     (envelope body point)

    - Computational Domain and Resolution Aspects: 
            
            SIZE: simulation setup for domain of size [0,4]x[0,1.5]
                 (smaller than domain used to collect NN training/test data)

            RESOLUTION: Nx=512, Ny=192 (dx=Lx/Nx=Ly/Ny=dy)
                 (simulation is half as resolved as those used to collect
                        NN training and test dataset)

            RATIONALE: this simulation only takes roughly ~35 minutes to run
                 (for this resolution, the time-step (dt) could also be increased 
                  compared to the smaller one used to collect data at higher resolutions)

            WHY ACTUAL SIMULATION TIME IS SHORTER?
                 Less grid cells (smaller Nx,Ny) and larger time-steps (bigger dt)
                 decrease the overall computational time to run a simulation 
                  but at the price of a less accurate simulation