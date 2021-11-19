
import numpy as np
import random


class SVM:
    # w:  a numpy array of weights (+b)
    def __init__(self, w):
        self.w = w
        
    # The gradient wrt w, used in Stochastic GD
    # x: a single data sample
    # y: the label of the sample, in the set {-1, 1}
    # C: tradeoff between regularization and empirical loss (recommended: 1/N)
    # N: Number of samples in the dataset
    # returns: gradient of the function wrt [b, w]
    def Gradient(self, x, y, C, N):
        w = self.w
        out = w.copy()
        out[0] = 0     # regularization
        loss = 1 - y * w.dot(x)
        if(loss > 0):  # hinge-loss
            out -= C * N * y * x
        return out
    
    # Returns the value of the SVM objective function
    def ObjFunc(self, x, y, C, N):
        w = self.w
        return 0.5 * w.dot(w) + C * N * max(0, 1-y*(w.dot(x)))
    
    # Performs an update
    # data is a single example
    # label is either -1 or 1
    # r: the learning rate
    # C: tradeoff between regularization and empirical loss (recommended: 1/N)
    # N: Number of samples in the dataset
    def Step(self, x, y, r, C, N):
        grad = self.Gradient(x, y, C, N)
        self.w -= r * grad
        
    # Predicts the label of an example
    # output is either -1 or 1
    def PredictLabel(self, data):
        return np.sign(data.dot(self.w))


# Runs the Perceptron algorithm for one of the models above
# C: tradeoff between regularization and empirical loss (recommended: 1/N)
# rfunc: a callable function of (t) to get the current learning rate
#        the learning rate updates at each epoch
# ReportObjFunc: If True, calculates the value of the objective function after each update
#                These values are returned at the end of the function
def SVMSGD(data, labels, model, T, C, rfunc, rargs, ReportObjFunc=False):
    N = data.shape[0]
    if ReportObjFunc:
        losses = []
    # generate a list of indexes to shuffle
    indexes = np.array(list(range(N)))
    for t in range(T):
        # shuffle the data at the start of the epoch
        np.random.shuffle(indexes)
        # update the learning rate
        r = rfunc(t, rargs)
        # run through each item in the dataset
        for i in indexes:
            x = data[i]
            y = labels[i]
            model.Step(x, y, r, C, N)
            if ReportObjFunc:
                losses.append(model.ObjFunc(x, y, C, N))
    if ReportObjFunc:
        return losses


class SVMDual:
    # w:  a numpy array of weights (+b)ADFASDF
    def __init__(self, w):
        self.w = w
        
    # The gradient wrt w, used in Stochastic GDASDFASDF
    # x: a single data sample
    # y: the label of the sample, in the set {-1, 1}
    # C: tradeoff between regularization and empirical loss (recommended: 1/N)
    # N: Number of samples in the dataset
    # returns: gradient of the function wrt [b, w]
    def Gradient(self, x, y, C, N):
        w = self.w
        out = w.copy()
        out[0] = 0     # regularization
        loss = 1 - y * w.dot(x)
        if(loss > 0):  # hinge-loss
            out -= C * N * y * x
        return out
    
    # Returns the value of the SVM objective functionASDFAF
    def ObjFunc(self, x, y, C, N):
        w = self.w
        return 0.5 * w.dot(w) + C * N * max(0, 1-y*(w.dot(x)))
    
    # Performs an updateASDFADF
    # data is a single example
    # label is either -1 or 1
    # r: the learning rate
    # C: tradeoff between regularization and empirical loss (recommended: 1/N)
    # N: Number of samples in the dataset
    def Step(self, x, y, r, C, N):
        grad = self.Gradient(x, y, C, N)
        self.w -= r * grad
        
    # Predicts the label of an exampleADFASFD
    # output is either -1 or 1
    def PredictLabel(self, data):
        return np.sign(data.dot(self.w))


# Finds the average error of a perceptron model over a dataset
def AverageError(data, labels, model):
    N = data.shape[0]
    outputs = model.PredictLabel(data)
    incorrectoutputs = [1 if outputs[i] != labels[i] else 0 for i in range(N)]
    return np.sum(incorrectoutputs) / N
