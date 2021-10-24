
from LinearRegression.Linear import *
import numpy
import random


# Performs the gradient descent algorithm 
# data: the rows of data; must be a 2-dim numpy array
# labels:  the labels corresponding to the rows of data
# classifier:  A classifier object with weights, bias, and a loss function
# batchsize:  the batchsize for batch g.d.;  batchsize=1 => stochastic g.d.
# r:          the learning rate
# tolerance:  The largest change in w where convergence is considered
# T:          max number of iterations; T=None => no limit
def GradientDescent(data, labels, classifier, batchsize, r, tolerance=None, T=None, 
                    giveTrainingLoss=False, giveTestingLoss=False, testingdata=None, testinglabels=None):
    # Get initial loss
    if giveTrainingLoss:
        trlosses = []
        trlosses.append(classifier.Loss(data, labels))
    if giveTestingLoss:
        tstlosses = []
        tstlosses.append(classifier.Loss(testingdata, testinglabels))
    # Set up variables
    curidx = 0 if batchsize > 1 else random.randrange(data.shape[0])
    endidx = batchsize  # the ending index doesn't have to be in the bounds of the array
    # Perform g.d.
    while True:
        # remember the old weights
        oldw = classifier.w.copy()
        # calculate the gradient
        grad = classifier.Gradient(data[curidx:endidx], labels[curidx:endidx])
        # update the weights
        classifier.w -= r * grad
        # check the loss
        if giveTrainingLoss:
            trlosses.append(classifier.Loss(data, labels))
        if giveTestingLoss:
            tstlosses.append(classifier.Loss(testingdata, testinglabels))
        # check for tolerance
        if tolerance != None and np.average(np.abs(classifier.w - oldw)) <= tolerance:
            print('stopped due to tolerance')
            break
        # check for iterations (if there is a max iterations)
        if T != None:
            if T <= 1:
                print('ran out of iterations')
                break
            else:
                T -= 1
        # update the batch indeces
        curidx = endidx % data.shape[0] if batchsize > 1 else random.randrange(data.shape[0])
        endidx = curidx + batchsize
    # return the classifier and report the loss (if requested)
    if giveTrainingLoss and giveTestingLoss:
        return (classifier, trlosses, tstlosses)
    if giveTrainingLoss:
        return (classifier, trlosses)
    if giveTestingLoss:
        return (classifier, tstlosses)
    else:
        return classifier
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        