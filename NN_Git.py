# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 00:46:30 2020

@author: Riccardo
"""

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import RobustScaler
from scipy.stats import truncnorm




#%%

Input_Data = pd.read_csv("C:\\Users\\Riccardo\\Documents\\PhD\\ML Research Topics\\Deep Learning Research\\ionosphere.csv")

Y_Train = np.array(Input_Data.iloc[:,-1])

X_Train = np.array(Input_Data.iloc[:,0:Input_Data.shape[1]-1])



print(X_Train)
print(Y_Train)


X_Train = X_Train[0:170,:]
Y_Train = Y_Train[0:170]

Scaler = RobustScaler()
Scaler.fit(X_Train)
X_Train = Scaler.transform(X_Train)


#%%

#Play inputs:

X_Train = np.array(([10,20,11,12,12],[1,2,1,2,2], [1,3,4,2,5], [10,20,11,12,12]))#rows observations, columns variables
Y_Train = [1,0,0,1]

X_Test = [1,2,3]
Y_Test = [0,0,0]

Shape = [2,5,10]

#Index information:
#rows of input data are observations, columns are attributes

#%%


def Weights_Inizialization(Shape, X_Train):
    
    Shape_Full = [X_Train.shape[1]] + Shape + [1]
    
        
    W_Array = [] #this contains the weights across layers
    B_Array = []
    for j in range(0,len(Shape_Full)-1):
        W_Layer = []
        B_Layer = []
        for i in range(0,Shape_Full[j+1]): #this figures out the weights for a single layer, j+1 cause there will be j+1 pairs of layers, meaning if I have 5 input dimension and next layer is 2 neurons, then I need 5 weights to multiply the input into the first next layer neuron, and 5 other weights to multiply the input into the second next layer neuron.
            
            values = np.random.normal(0, 1, size = Shape_Full[j])
            
            W = values
            #truncnorm.rvs(0, 0.05, size = Shape_Full[j])
            W_Layer.append(W)
            
        B = 0 
        B_Layer.append(B)#inserted in this loop so that list index will be consistent with weight array
        W_Array.append(W_Layer)
        B_Array.append(B_Layer)
    
    #W_Array = np.matrix(np.array(W_Array))
    #B_Array = np.matrix(np.array(B_Array))

    return W_Array, B_Array

Weights_Inizialization(Shape, X_Train)



###figure out b array issue

#%%

def ReLU(Input):
    
    Input[Input<0] = 0
    
    return Input

def Logit(Input):
    
    for i in range(0,len(Input)):
            
        Input[i] = 1/(1+math.exp(-Input[i])) #check if correct
        
    return Input



def ReLU_Grad(Input):
    
    Input[Input == 0] = 0
    Input[Input != 0] = 1
    
    return Input

def Logit_Grad(Input):

    for i in range(0,len(Input)):
        
        try:
                
            Input[i] = math.exp(Input[i])/((1+math.exp(Input[i]))**2)

        except OverflowError:
    
            Input[i] = 0
    
            
    return Input


#%%




def Forward_NN(Shape, X_Train, Y_Train, learning_rate):
    
    W_Array, B_Array = Weights_Inizialization(Shape, X_Train)
    
    #print(W_Array[0])
    
    #number of epochs (full forward and backward training)
    for epoch in range(0,1200):

        
        ###FORWARD PASS:
        #initial layer:
        Z_Array = [] #raw outputs per layer
        A_Array = [] #final activation outputs per layer
        
        Layer_Output = np.matrix(X_Train)*(np.matrix(W_Array[0]).T) + B_Array[0] #matrix multiplication of inputs by first set of weights, diagram on excel; then add the bias term
        Z_Array.append(Layer_Output)
        A_Array.append(ReLU(Layer_Output))#activation for the previous z output
                
        #same as above but now iterates to multiply each preceding layer by each subsequent set of weights
        for i in range(0,len(Shape)):         
            
            Layer_Output = np.matrix(Z_Array[i])*(np.matrix(W_Array[i+1]).T) + B_Array[i+1]
            
            Z_Array.append(Layer_Output)
            A_Array.append(ReLU(Layer_Output))
        

        try:
            Y_Prediction = Logit(np.array(Z_Array[-1])) #logit of the last layer in the z array list of lists
        except OverflowError:
            Y_Prediction = 0 #0 if overflow error
        
        
        Y_Prediction[Y_Prediction == 1] = 0.99#clip values that are exact 1 or exact 0 if they're ever reached to allow cross entropy function not to explode
        Y_Prediction[Y_Prediction == 0] = 0.01
        
        
        #cross entropy loss
        Error = np.array(Y_Train)*np.log(np.array(Y_Prediction.transpose())) + (1-np.array(Y_Train))*np.log(1-np.array(Y_Prediction.transpose()))
        
        Mean_Error = Error.mean()
        
        print("Cross Entropy Error: ", Mean_Error)
        #print("Predicted: ", Y_Prediction)
        #print("Real: ", Y_Train)
    
        ###BACKWARD PASS:
        
        Mean_Weight_Gradients = []
        Mean_Bias_Gradients = []
        
        
        #run loop that does backprop for each observation individually
        for obs_index in range(0,len(X_Train)):
            
            Backprop_Array = [] #cumulative backpropagation at each layer
            
            Weight_Gradients = []
            
            Bias_Gradients = []
    
            #first append derivative of y (logit derivative at last z array value[observation]) wrt last activation; so dy/da = dz/da (last weights)*dy/dz (logit grad at z)
            
            Error_Gradient = -((Y_Train[obs_index]/Y_Prediction[obs_index]) - ((1-Y_Train[obs_index])/(1-Y_Prediction[obs_index])))
            
            
        
            Backprop_Array.append(np.matrix(np.matmul(Logit_Grad(np.matrix(Z_Array[-1][obs_index]))*Error_Gradient,W_Array[-1])))
            
            Weight_Gradients.append(np.matrix(np.matmul(Logit_Grad(np.matrix(Z_Array[-1][obs_index]))*Error_Gradient,np.matrix(Z_Array[-2][obs_index]))))

            Bias_Gradients.append(Logit_Grad(np.matrix(Z_Array[-1][obs_index])))
            

            for j in range(0,len(Shape)-1): #for each layer of the network do backprop, len-1 because we alreadyd did the first layer out of the loop above
                
                                
                Backprop_as_of_now = np.multiply((ReLU_Grad(np.matrix(Z_Array[-j-2][obs_index]))), Backprop_Array[-1]) #index is -j-2 for z array cause already did -1 outside of loop
                #above is step to do element by element (not matrix) multiplication of same shape arrays for cumulative backprop and gradient at z input (this being dA/dZ)
                                
                Backprop_Array.append(Backprop_as_of_now)
                #update latest cumulative backprop
                
                Bias_Gradients.append(Backprop_as_of_now)
                
                if j == len(Shape)-1:
                    
                    break
                
                else:
                    
                    Weight_Gradients.append(np.matmul(Backprop_as_of_now.transpose(),np.matrix(Z_Array[-j-3][obs_index]))) #before doing next step calculate weight gradient wrt to prior input layer z instead of weight in next lines of code
                    
                    Backprop_as_of_now = np.matmul(Backprop_Array[-1], np.matrix(W_Array[-j-2])) #now matrix (not element by element) multipllication of different but cohesive shape arrays, one being cumulative backprop and other being the weights or dZ/dA
                    
                    Backprop_Array.append(Backprop_as_of_now)
                    
            
            
            Weight_Gradients.append(np.multiply(Backprop_as_of_now.transpose(),np.matrix(X_Train[obs_index,:]))) #now do final gradient wrt to first set of weights in first layer, and need to multiply cumulative backprop by x input which is not in z array

            Bias_Gradients.append(np.multiply((ReLU_Grad(np.matrix(Z_Array[0][obs_index]))), Backprop_Array[-1]))
                        
        
            if len(Mean_Weight_Gradients) == 0:
                
                
                Mean_Weight_Gradients.append((1/len(X_Train))*np.array(Weight_Gradients)) #incremental mean of gradients
                                
                for element in Bias_Gradients:
                    element = element*(1/len(X_Train))
                
                Mean_Bias_Gradients.append(Bias_Gradients)
    
            else:
                                
                Update = Mean_Weight_Gradients[0] + ((1/len(X_Train))*np.array(Weight_Gradients))
                
                Mean_Weight_Gradients[0] = Update
            
                
                #biases:
                
                for index in range(0,len(Bias_Gradients)):
                    
                    Mean_Bias_Gradients[0][index] = Mean_Bias_Gradients[0][index] + ((1/len(X_Train))*Bias_Gradients[index])
                                                
        
        Mean_Weight_Gradients = Mean_Weight_Gradients[0] #previously we did an array of the entire thing in the if statement so this takes that out so that [i] of this is a single layer's gradients for a single layer's weights
        Mean_Bias_Gradients = Mean_Bias_Gradients[0]
        
    
        ######GRADIENT DESCENT WEIGHT UPDATE:
        
        for i in range(0,len(W_Array)):
                                                
            W_Array[i] = np.matrix(W_Array[i]) - learning_rate*np.matrix(Mean_Weight_Gradients[len(W_Array)-1-i])#mean weight gradients list has inverse order/indexing as weight array, that's why index different here

            B_Array[i] = np.matrix(B_Array[i]) - learning_rate*np.matrix(Mean_Bias_Gradients[len(B_Array)-1-i])
    
    

        Y_Prediction[Y_Prediction > 0.5] = 1
        Y_Prediction[Y_Prediction <= 0.5] = 0
        
        Errors = 0
        for i in range(0,len(Y_Train)):
            
            if Y_Prediction[i] != Y_Train[i]:
                Errors = Errors + 1
        
        Accuracy = ((len(Y_Train)-Errors)/len(Y_Train))*100
        print("Model Accuracy: ", Accuracy, "%")
    
    
    #Final Accuracy:
    
    Y_Prediction[Y_Prediction > 0.5] = 1
    Y_Prediction[Y_Prediction <= 0.5] = 0
    
    
    Errors = 0
    for i in range(0,len(Y_Train)):
        
        if Y_Prediction[i] != Y_Train[i]:
            Errors = Errors + 1
    
    Accuracy = ((len(Y_Train)-Errors)/len(Y_Train))*100
    print("Model Accuracy: ", Accuracy, "%")
    
    
    return Z_Array


#%%
    
#RUN MODEL:

Output = Forward_NN(Shape, X_Train, Y_Train, 0.01)

print("Final output:", Output)



#%%

#do optimizer that draws from RL, with 5% chance weight update is random, and 95% chance it's gradient descent
#hyperparameter tuning with the above or just simple gradient descent
#use neural network to predict hyperparameter values, as in train on hyperparameter raedings as X and loss as Y and understand loss function with NN, then predict minimum loss
#try new activation function
#apply gradient descent to weight initialization, can include some metrics for sparsity and whatever and get gradient wrt those


#%%

#sanity check

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras import backend as K
import tensorflow as tf

from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1


initializer = tf.keras.initializers.GlorotUniform()
values = np.array(initializer(shape=(1,Shape_Full[j])))[0]
print(values)
#%%

model = Sequential()

model.add(Dense(units = Shape[0], activation='relu', bias_initializer='zeros', input_dim = (np.shape(X_Train)[1])))

model.add(Dense(units = Shape[1], activation='relu'))

model.add(Dense(units = Shape[2], activation='relu'))

model.add(Dense(units = 1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = Adam(lr = 0.001), metrics=['accuracy'])

model.fit(X_Train, Y_Train, epochs = 800, batch_size = len(X_Train), shuffle = False, verbose = 1)

Validation_Performance = model.evaluate(X_Train, Y_Train, verbose=1, batch_size = len(X_Train))


print(Validation_Performance)
