
import numpy as np

# All Perceptron Classes must have the following:
# - Step(data, label):  Performs a step in the perceptron algorithm
# - PredictLabel(data):  Makes a prediction of the label of the datapoint

class PerceptronStandard:
    # w:  a numpy array of weights (+b)
    # r:  the learning rate
    def __init__(self, w, r):
        self.w = w
        self.r = r
        
    # Performs an update
    # data is a single example
    # label is either -1 or 1
    def Step(self, data, label):
        pred = self.PredictLabel(data)
        if pred != label:
            self.w += label * self.r * data
            
    # Predicts the label of an example
    # output is either -1 or 1
    def PredictLabel(self, data):
        return np.sign(data.dot(self.w))
        
        
class PerceptronVoted:
    # w:  a numpy array of weights (+b)
    # r:  the learning rate
    def __init__(self, w, r):
        self.w = w
        self.r = r
        self.c = 0
        # a list of tuples (w, c_w)
        self.wlist = []
        
    # Performs an update
    # data is a single example
    # label is either -1 or 1
    def Step(self, data, label):
        pred = np.sign(self.w.dot(data))
        if pred != label:
            self.wlist.append((self.w.copy(), self.c))
            self.w += label * self.r * data
            self.c = 0
        else:
            self.c += 1
            
    # Predicts the label of an example
    # output is either -1 or 1
    def PredictLabel(self, data):
        out = np.zeros(data.shape[0])
        for (w, c) in self.wlist:
            out += np.sign(data.dot(w)) * c
        return np.sign(out)



class PerceptronAveraged:
    # w:  a numpy array of weights (+b)
    # r:  the learning rate
    def __init__(self, w, r):
        self.w = w
        self.r = r
        self.a = np.zeros(w.shape)
        
    # Performs an update
    # data is a single example
    # label is either -1 or 1
    def Step(self, data, label):
        pred = np.sign(self.w.dot(data))
        if pred != label:
            self.w += label * self.r * data
        self.a += self.w
        
    # Predicts the label of an example
    # output is either -1 or 1
    def PredictLabel(self, data):
        return np.sign(data.dot(self.a))


# Runs the Perceptron algorithm for one of the models above
def RunPerceptron(data, labels, pmodel, T):
    # generate a list of indexes to shuffle
    indexes = np.array(list(range(data.shape[0])))
    for t in range(T):
        # shuffle the data at the start of the epoch
        np.random.shuffle(indexes)
        # run through each item in the dataset
        for i in indexes:
            pmodel.Step(data[i], labels[i])


# Finds the average error of a perceptron model over a dataset
def AverageError(data, labels, pmodel):
    numrows = data.shape[0]
    outputs = pmodel.PredictLabel(data)
    incorrectoutputs = [1 if outputs[i] != labels[i] else 0 for i in range(numrows)]
    return np.sum(incorrectoutputs) / numrows





















