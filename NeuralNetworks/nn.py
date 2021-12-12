
import numpy as np
import scipy
from scipy.special import expit as sigmoid

# initializes the weights and biases for three layers of the three-layer neural netork with the normal distribution.
# Returns (w_0, b_0, w_1, b_1, w_f, b_f)
def initnormal(layerlength, datalength):
    w_0 = np.random.normal(size=(layerlength, datalength))
    w_1 = np.random.normal(size=(layerlength, layerlength))
    w_f = np.random.normal(size=(layerlength))
    (b_0, b_1, b_f) = np.random.normal(size=(3))
    return (w_0, b_0, w_1, b_1, w_f, b_f)

# initializes the weights and biases for three layers of the three-layer neural netork with 0.
# Returns (w_0, b_0, w_1, b_1, w_f, b_f)
def initzeros(layerlength, datalength):
    w_0 = np.zeros((layerlength, datalength))
    w_1 = np.zeros((layerlength, layerlength))
    w_f = np.zeros(layerlength)
    b_0 = 0
    b_1 = 0
    b_f = 0
    return (w_0, b_0, w_1, b_1, w_f, b_f)
    
# initializes the weights and biases for three layers of the three-layer neural netork with 1.
# Returns (w_0, b_0, w_1, b_1, w_f, b_f)
def initones(layerlength, datalength):
    w_0 = np.ones((layerlength, datalength))
    w_1 = np.ones((layerlength, layerlength))
    w_f = np.ones(layerlength)
    b_0 = 0
    b_1 = 0
    b_f = 0
    return (w_0, b_0, w_1, b_1, w_f, b_f)

 # Calculates the derivative of the sigmoid function with respect ot its input
def der_sigmoid(a):
    return np.exp(-a) * sigmoid(a)
        
# A three-layer binary-classification neural network
# datalength: the dimensionality of a data sample
# Layerwidth: the number of neurons in each hidden layer
# initfunc: given a shape (layer_length, data_length), initializes the weights and biases
class ClassifNN:
    def __init__(self, datalength, layerwidth, initfunc):
        # Weights for a layer are a matrix in this structure:
        # [weights for neuron 1]
        # [weights for neuron 2]
        # ...
        # Biases are a separate np.array
        # A pass through a single layer is the following:
        # activation(data.dot(w.T)+b)
        (self.w_0, self.b_0, self.w_1, self.b_1, self.w_f, self.b_f) = initfunc(layerwidth, datalength)
    
    # Performs a step of stochastic gradient descent given a single data sample
    # data: a single data sample
    # label: the label for the data sample
    # lr: the learning rate for this step
    # reportloss: if true, returns the output of the loss function
    def gradientdescentstep(self, data, label, lr, reportloss=False):
        # Forward pass
        lin_0 = data.dot(self.w_0.T) + self.b_0
        out_0 = sigmoid(lin_0)
        lin_1 = out_0.dot(self.w_1.T) + self.b_1
        out_1 = sigmoid(lin_1)
        out_f = out_1.dot(self.w_f.T) + self.b_f
        # loss
        loss = 0.5 * (out_f - label)**2
        
        # Backward pass
        # Calculate derivatives from the loss to some point
        der_outf = out_f - label
        
        der_wf = der_outf * out_1
        der_bf = der_outf * 1
        der_out1 = der_outf * self.w_f
        
        der_lin1 = der_out1 * der_sigmoid(lin_1)
        der_w1 = np.outer(der_lin1, out_0)
        der_b1 = der_lin1 * 1
        der_out0 = der_lin1.dot(w_1)
        
        der_lin0 = der_out0 * der_sigmoid(lin_0)
        der_w0 = np.outer(der_lin0, data)
        der_b0 = der_lin0 * 1
        
        # Apply gradients
        self.w_f -= lr * der_wf
        self.w_b -= lr * der_bf
        self.w_1 -= lr * der_w1
        self.w_1 -= lr * der_b1
        self.w_0 -= lr * der_w0
        self.w_0 -= lr * der_b0
        
        # Report the loss, if needed
        if reportloss:
            return loss
            
    
    # Performs stochastic gradient descent on the model
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    