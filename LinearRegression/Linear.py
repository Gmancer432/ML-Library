
import numpy as np


# A linear model used in linear regression/classification
class LMS:
    # w must be a numpy array, where w[0] = b
    def __init__(self, w):
        self.w = w
        
    # data must be a numpy array
    def PredictLabel(self, data):
        return data.dot(self.w)
    
    # Gives LMS loss
    # Labels must be a numpy array
    # x[:][0] = 1
    def Loss(self, data, labels):
        return 0.5 * (((labels - self.PredictLabel(data))**2).sum())
        
    # Returns the gradient of LMS wrt w, given x
    # x[:][0] = 1
    def Gradient(self, x, y):
        out = -1 * (((y - x.dot(self.w)) * (x.T)).T)
        if x.ndim > 1:
            out = out.sum(axis=0)
        return out


# If data is linearly separable, this will find the optimal weights
def AnalyticalLMS(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)