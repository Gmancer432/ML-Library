
import numpy as np
import random
import scipy.optimize
import scipy.spatial.distance


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


class LinearKernel:
    def __init__(self, args):
        pass
    
    # Vectorized form used during optimization
    def optim(self, x, y, a):
        tocross = (x.T * y * a).T
        return tocross.dot(tocross.T).sum()
    
    # Vectorized form used during prediction
    def pred(self, x, y, a, newx):
        tocross = (x.T * y * a).T
        return tocross.dot(newx.T).sum(axis=0)
    
    
class GaussianKernel:
    def __init__(self, gamma):
        self.gamma = gamma
    
    # Vectorized form used during optimization
    def optim(self, x, y, a):
        # compute the squared distance between all pairs of vectors in x
        dists = scipy.spatial.distance.cdist(x, x, 'sqeuclidean')
        # prepare matrix of scalars
        crossedscalars = np.outer(y*a, y*a)
        # complete the gaussian kernal + sum
        return (crossedscalars * np.exp(-(dists/self.gamma))).sum()
    
    # Vectorized form used during prediction
    def pred(self, x, y, a, newx):
        # compute the squared distance between all pairs of vectors in x, newx
        # These are in the form x_1,newx_1; x_1,newx_2; ...
        #                       x_2,newx_1; x_2,newx_2; ...
        #                       ...
        dists = scipy.spatial.distance.cdist(x, newx, 'sqeuclidean')
        # finish the kernal
        kout = np.exp(-dists/self.gamma)
        # scale and sum
        return (kout.T * y * a).T.sum(axis=0)


class SVMDual:

    def __init__(self, kernel=LinearKernel, kernelargs=None):
        self.svlim = 10E-10
        self.svdata = None
        self.svlabels = None
        self.svastar = None
        self.b = 0
        self.kernel = kernel(kernelargs)
    
    # The function to minimize (within the constraints)
    def dualfunc(self, a, data, labels, kernel):
        return 0.5 * kernel.optim(data, labels, a) - a.sum()
    
    # Finds the optimal solution, given a set of data and labels
    # Unlike in the primal SVM, this data is NOT augmented
    # C: tradeoff between regularization and loss (recommended: 1/N)
    def Optimize(self, data, labels, C):
        N = data.shape[0]
        a = np.zeros(N)
        # Define the bounds for optimization
        abounds = scipy.optimize.Bounds(0, C)  # 0 <= a <= C
        linearconstraint = scipy.optimize.LinearConstraint(labels, 0, 0) # sum(a * y) = 0
        # Optimize
        args = (data, labels, self.kernel)
        astar = scipy.optimize.minimize(self.dualfunc, a, args=args, bounds=abounds, constraints=linearconstraint).x
        # count up the support vectors and store them
        tokeep = astar > self.svlim
        self.svdata = data[tokeep]
        self.svlabels = labels[tokeep]
        self.svastar = astar[tokeep]
        # retrieve b
        self.b = np.average(self.svlabels - self.PredictLabel(self.svdata))
        
    
    # Returns the value of the SVM objective function
    def ObjFunc(self, x, y, C, N):
        raise NotImplementedError
        
    # Predicts the label of an example
    # Unlike in the primal SVM, input data is NOT augmented with 1
    # output is either -1 or 1
    def PredictLabel(self, data):
        return np.sign(self.kernel.pred(self.svdata, self.svlabels, self.svastar, data) + self.b)


# Finds the average error of a perceptron model over a dataset
def AverageError(data, labels, model):
    N = data.shape[0]
    outputs = model.PredictLabel(data)
    incorrectoutputs = [1 if outputs[i] != labels[i] else 0 for i in range(N)]
    return np.sum(incorrectoutputs) / N
