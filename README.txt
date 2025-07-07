---------------------------------------------------------------------------                               
                               README.txt

        --> Information regarding folders (and subfolders) within 
            main github repository

---------------------------------------------------------------------------

FOLDERS:

    [1] NN_Training: "all NN Training codes, including the training and 
                      test datasets and data transformation script"

        Included scripts:

            --> go_Go_NN_Training.m: 
                    - MAIN code to run to perform NN training
                    - Where you set NN hyperparameters for training

            --> Train_Artificial_Neural_Network.m:
                    - Script that actually performs NN training
                    - Is called from the go_Go_NN_Training.m script)

            --> scale_The_Output_Data.m:
                    - Transforms the Model Output for NN performance using 
                          (i) linear transformation
                         (ii) Box-Cox (power transform)
                        (iii) Standardization (z-Transformation)
                    - Is called from the go_Go_NN_Training.m script

            --> forward_propagate.m: 
                    - performs Forward Propagation on given input data
                    - each input parameter combination must be in column 
                      vector form
                    - will get used whenever using NN for prediction

            --> activation_function.m && act_function_PRIME.m:
                    - activation function and its derivative
                    - Note: set to ReLU by default; if switching to Sigmoid,
                      may need to SUBSTANTIALLY revise hyperparameter selection
                                    
    [2] NN_Prediction: codes that use the NN to prediction performance
            
        Included scripts:

            --> Prediction_Tamp_Freq.m:
                    - predicts performance across the (Tamp,f)-subspace
                      for constant values of (Hamp/Tamp, lambda, xB*)
                    - plots speed and power across the subspace
            
            --> Prediction_Errors.m:
                    - performs NN error analysis, i.e., how well the 
                      NN recovers the training dataset and predicts the 
                      test dataset.
                    - provides qualitative plots and error statistics
                
            --> scale_Input_Parameters_for_NN.m: 
                    - scales model input parameters from their actual real 
                      values to [-1,1] for NN prediction


    [3] Optimization:

            --> Basic_Gradient_Descent.m: 
                    - basic gradient descent code to find minimum of the 
                      function f(x,y) = 2*(x-1)^2 + 3*(y+2)^2  

            --> Best_Fit_Ex1_Data.mat:
                    - (X,Y) data samples to be used to find a best fit cubic
                      polynomial f(x;vec{c}) = c0 + c1*x + c2*x^2 + c3*x^3)

            --> Linear_Least_Squares_Best_Fit.m:
                    - uses Linear Least Squares to find the best fit 
                      cubic polynomial through a provided dataset from
                      Best_Fit_Example_Data.mat

            --> Gradient_Descent_Best_Fit.m:
                    - uses standard gradient descent to find the best fit 
                      cubic polynomial through a provided dataset from
                      Best_Fit_Example_Data.mat

            --> Stochastic_Gradient_Descent_Best_Fit.m:
                    - uses stochastic gradient descent to find the best fit 
                      cubic polynomial through a provided dataset from
                      Best_Fit_Example_Data.mat

           --> Best_Fit_Ex2_Data.mat:
                    - (X,Y) data samples to be used to find a best fit function
                      of the following form: f(x;vec{c}) = c0 + c1*sin(2x) + c3*e^(x/4)


    [4] CFD Swimmer:
            
        Included subfolders (more details provided in CFD_Swimmer/README.txt):

            --> SUBFOLDER: Single_Swimmer_Sim/
        
                    - scripts needed to perform a single swimmer simulation
                    - to run the simulation, run the main2d.m file

            --> SUBFOLDER: Analyze_Single_Sim_Data/
    
                    - scripts to analyze and visualize a simulation's data

                    - Swimmer_Analysis.m: 
                            - loads simulation data to plot performance metrics vs. time
                              (ex: speed vs time, power vs time, etc.)

                    - Swimmer_Visualization.m:
                            - visualizes the swimmer's geometry and fluid data as
                              as the simulation progresses (ie, shows it swimming)
                            - default simulation saves vorticity data (Omega) and that
                              is the fluid data visualized here. To save other fluid data,
                              select that desired data to save in the input2d file.

            --> SUBFOLDER: IBM_Blackbox_REVISED_FEB2024: "black box scripts that run 
                                                  an immersed boundary simulation"
    
                    - YOU SHOULD NOT HAVE TO MODIFY THIS SCRIPTS! :)

                    - To have a better idea of what they're doing, I suggest reading 
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

                    - In addition, you can also look into overvies of immersed boundary methods,
                      such as Peskin 2002:

                        (a) Peskin, C.S., The immersed boundary method, Acta numerica, 11:479-517 (2002).

                                                            