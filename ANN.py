# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:27:39 2021

@author: DELL
"""

import  numpy as np


class Neuron():
    def __init__(self,no_of_inputs):
        self.bias= 0.2
        self.weights=[0.65,0.30]
        self.learning_rate=0.1
    def predicit(self,input1,input2):
        summation = input1*self.weights[0]  + input2*self.weights[1] + self.bias
        if summation >=0:
            return 1
        else:
            return 0
    for epoch in range(20):
        for items in range(100):
           def  train(self,training_ex,actual_label):
               predicted_label= self.predicit(training_ex[0], training_ex[1])
               print("predicted_label  is:",predicted_label,"bias is:",self.bias)
               
               if(predicted_label!=actual_label):
                   error = actual_label  -  predicted_label
                   delta_w = self.learning_rate * error
                   self.weights[0] += delta_w
                   self.weights[1] += delta_w
                   self.bias +=delta_w
                   print("error is:", error,"new bias is:", self.bias ,"predicted_label is:",predicted_label)
               return actual_label


   
x1=0
x2=0
y=0

training_input = np.array([x1,x2])
neuron =  Neuron(2)
decision =  neuron.train(training_input, y)
print("The decision is:", decision)







