---------------------------------------------------------------------------                               
                                README
---------------------------------------------------------------------------


HOW TO RUN A SIMULATION: 

    - run the main2d.m script in the Single_Swimmer_Sim/ folder

HOW TO ANALYZE AND VISUALIZE DATA THAT IS PRODUCED: 

    - run the Swimmer_Analysis.m and Swimmer_Visualization.m scripts, 
      respectively in the Analyze_Single_Sim_Data/ folder

    - NOTE: can visualize .vtk data that gets produced using open-source
      visualization software, such as VisIt: 
      https://visit-dav.github.io/visit-website/index.html

FOLDERS:

    [1] Single_Swimmer_Sim: "scripts needed to perform a single swimmer simulation"
                            "to run the simulation, run the main2d.m file"

        Included scripts:

            --> main2d.m: 
                    - MAIN code to run the swimmer simulation
                    - Where you set NN hyperparameters for training

            --> input2d:
                    - provides inputs necessary for an IB simulation
                      (eg, properties of fluid, grid configuration, etc.)

            --> IBM_Driver_SWIMMER:
                    - main IB simulation driver file; this is where 
                      simulation is configured and run

            --> swimmer.vertex:
                    - gives all (X,Y) points along swimmer geometry at time 0

            --> swimmer.spring:
                    - describes all spring connections between node IDs and             
                      gives spring properties (stiffness, resting length)

            --> swimmer.nonInv_Beam:
                    - describes all beam connections between node IDs and             
                      gives beam properties (stiffness, preferred curvatures)

            --> swimmer.target:
                    - describes all target points along node IDs and their
                      corresponding target stiffnesses

            --> swimmer_X.phases, swimmer_Y.phases:
                    - matrices describing all points along the swimmer (column)
                      at different interpolation time periods (rows) throughout
                      one stroke cycle

            --> update_nonInv_Beams.m, update_Springs.m, update_Target_Point_Positions.m:
                    - files that update properties regarding the beams, springs, 
                      and target points as the simulation progresses

    [2] Analyze_Single_Sim_Data: "scripts to analyze & visualize a simulation's data"

        Included scripts:
            
            --> Swimmer_Analysis.m: 
                    - loads simulation data to plot performance metrics vs. time
                      (ex: speed vs time, power vs time, etc.)

            --> Swimmer_Visualization.m:
                    - visualizes the swimmer's geometry and fluid data as
                      as the simulation progresses (ie, shows it swimming)
                    - default simulation saves vorticity data (Omega) and that
                      is the fluid data visualized here. To save other fluid data,
                      select that desired data to save in the input2d file.


    [3] IBM_Blackbox_REVISED_FEB2024: "black box scripts that run an immersed 
                                                         boundary simulation"
        Included scripts:
                    
            --> YOU SHOULD NOT HAVE TO MODIFY THIS SCRIPTS! :)

            --> These scripts come from the open-source immersed boundary software,
                IB2d: https://github.com/nickabattista/IB2d

            --> To have a better idea of what they're doing, I suggest reading 
                some of the papers surrounding the IB2d Software:

                (a) N.A. Battista, A.J. Baird, L.A. Miller, A mathematical model 
                    and MATLAB code for muscle-fluid-structure simulations, 
                    Integ. Comp. Biol. 55(5):901-911 (2015)

                (b) N.A. Battista, W.C. Strickland, L.A. Miller, IB2d: a Python 
                    and MATLAB implementation of the immersed boundary method, 
                    Bioinspiration and Biomemetics 12(3): 036003 (2017)

                (c) N.A. Battista, W.C. Strickland, A. Barrett, L.A. Miller, 
                    IB2d Reloaded: a more powerful Python and MATLAB implementation 
                    of the immersed boundary method, Math. Method. Appl. Sci. 41:8455-8480 (2018)                

            --> In addition, you can also look into overvies of immersed boundary methods,
                such as Peskin 2002:

                (a) Peskin, C.S., The immersed boundary method, Acta numerica, 11:479-517 (2002).
                    
                              