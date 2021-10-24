
from Util.util import *
import random

# A bagged classifier, created by a bagging algorithm (below)
class BaggedClassif:
	def __init__(self, classifs):
		self.classifs = classifs
	def PredictLabel(self, data):
		votes = {}
		for i in range(len(self.classifs)):
			label = self.classifs[i].PredictLabel(data)
			if label not in votes.keys():
				votes[label] = 0
			votes[label] += 1
		return max(votes, key=votes.get)


# Runs through the data, generates labels, and calculates the error of the predicions
def CalculateError(data, labelidx, classif):
	total = 0
	incorrect = 0
	for d in data:
		w = 1 
		total += w
		outlabel = classif.PredictLabel(d)
		if outlabel != d[labelidx]:
			incorrect += w
	return incorrect / total


# Calculates the average bias and variance of a learner over the given data
def BiasVariance(data, labelidx, classifs, poslabel):
	sumbias = 0
	sumvariance = 0
	for d in data:
		# Collect outputs
		outs = []
		sumout = 0
		for c in classifs:
			out = 1 if c.PredictLabel(d) == poslabel else -1
			outs.append(out)
			sumout += out
		meanout = sumout / len(classifs)
		# Bias
		sumbias += (meanout - (1 if d[labelidx] == poslabel else -1))**2
		# Variance
		sumdiff = 0
		for o in outs:
			sumdiff += (o - meanout)**2
		sumvariance += sumdiff / (len(classifs) - 1)
	# Average over all data and return
	return (sumbias / len(data), sumvariance / len(data))
		
		
			
			


# Takes a classifier and trains using a bagging method
# data:        the input data
# labelidx:    the index of the example's label in the row
# learner:     the name of a method that calls the learner
# lparams:     the params to be given to the learner (Excludes data and weightidx, the first two terms in the learner params)
# T:           the number of iterations to run
# m:           the number of examples to sample from the training set
# giveErrors:  True if the classifier should be tested at each iteration, and it's errors reported.
# testdata:    Testing data for calculating test error.
# Returns:     Returns an AdaBoosted classifier.
#              If giveErrors=True, also returns a list of T rows containing the following for each iteration:
#               Final classifier training error
#               Final classifier testing error (None if no testing data is given)
#               single classifier t training error
#               single classifier t testing error (None if no testing data is given)
def bagging(data, labelidx, learner, lparams, T, m, giveFinalErrors=False, giveSingleErrors=False, testdata=None):
	# Initialize the values
	baggedclassif = BaggedClassif([])
	weightidx = AddColumn(data, 1)
	errs = []
	
	# Go through the iterations
	for i in range(T):
		# Sample from the training data
		cursamples = random.choices(data, k=m)
		# Generate a single classifier
		curclassif = learner(cursamples, weightidx, *lparams)
		baggedclassif.classifs.append(curclassif)
		# Calculate errors
		curerrs = []
		if giveFinalErrors:
			curerrs.append(CalculateError(data, labelidx, baggedclassif))          # ftrainerr
			if testdata != None:
				curerrs.append(CalculateError(testdata, labelidx, baggedclassif))  # ftesterr
		if giveSingleErrors:
			curerrs.append(CalculateError(data, labelidx, curclassif))             # strainerr
			if testdata != None:
				curerrs.append(CalculateError(testdata, labelidx, curclassif))     # stesterr
		if len(curerrs) > 0:
			errs.append(curerrs)
			
	# Return the full classifier, and errors (if requested)
	if len(errs) == 0:
		return baggedclassif
	else:
		return (baggedclassif, errs)
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		