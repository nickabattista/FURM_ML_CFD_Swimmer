---------------------------------------------------------------------------                               
                           README_NN_Training.txt
---------------------------------------------------------------------------

HOW TO RUN: 
    - run the go_Go_NN_Training.m script to training the NN!

PRODUCES: 
    - Folder: Trained_Neural_Network_Weights/
            --> Contains 'optimized' weight & biases for NN prediction

NN ARCHITECTURE:

    - MULTIPLE INPUT neurons
    - MULTIPLE OUTPUT neurons
    - 2 HIDDEN LAYERS
    - Bias in every hidden layer
    - ReLU activation functions

NN HYPERPARAMETERS (for use in Train_Artificial_Neural_Network.m):

    - num_hid_layer_neurons: # of neurons in hidden layer
    - numIter: # of iterations for Stochastic Gradient Descent
    - learning rate: step-size for Stochastic Gradient Descent ("lambda")
    - batch_size: for Stochastic Gradient Descent
    - coeff: coefficient for scaling initial guesses for W1, W2,...

            ---> FOR REGULARIZATION OF COST FUNCTION <---
    - regularize_Flag:  choose which regularization scheme to use (or none)
    - lam_regularize: regularization parameter for Regularization

                  --> "ADAPTIVE" LEARNING RATE <---
    - "every X epochs, decrease learning rate by certain factor"

DATA

    - Training_and_Test_Datasets_PAPER.mat:

          --> TRAINING_INPUTS: 1000x5 matrix of model inputs (real-values, Halton)
          --> TESTING_INPUTS:  2000x5 matrix of model inputs (real-values, Sobol)
                   ORDER: [Tamp, Hamp/Tamp, lambda, freq, xB*]
    
          --> TRAINING_OUTPUTS: 1000x2 matrix of model outputs 
          --> TESTING_OUTPUTS:  2000x2 matrix of model outputs 
                   ORDER: col-1: speed (raw), col-2: power (raw)

    - Training_and_Test_Datasets_FLIPPED.mat:

          --> TRAINING_INPUTS: 2000x5 matrix of model inputs (real-values, Sobol)
          --> TESTING_INPUTS:  1000x5 matrix of model inputs (real-values, Halton)
                   ORDER: [Tamp, Hamp/Tamp, lambda, freq, xB*]
    
          --> TRAINING_OUTPUTS: 2000x2 matrix of model outputs 
          --> TESTING_OUTPUTS:  1000x2 matrix of model outputs 
                   ORDER: col-1: speed (raw), col-2: power (raw)
 