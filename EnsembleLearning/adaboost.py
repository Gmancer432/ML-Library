
from Util.util import *
import math


# An Adaboost classifier, created by the adaboost algorithm (below)
class Adaboost:
	def __init__(self, a, classifs):
		self.a = a
		self.classifs = classifs
	def PredictLabel(self, data):
		votes = {}
		for i in range(len(self.a)):
			label = self.classifs[i].PredictLabel(data)
			if label not in votes.keys():
				votes[label] = 0
			votes[label] += self.a[i]
		return max(votes, key=votes.get)


# Runs through the data, generates labels, and calculates the error of the predicions
# weightidx=None if there are no weights
def CalculateError(data, labelidx, classif, weightidx):
	total = 0
	incorrect = 0
	for d in data:
		w = 1 if weightidx == None else d[weightidx]
		total += w
		outlabel = classif.PredictLabel(d)
		if outlabel != d[labelidx]:
			incorrect += w
	return incorrect / total


# Takes a classifier and trains using the AdaBoost method
# data:        the input data
# labelidx:    the index of the example's label in the row
# learner:     the name of a method that calls the learner
# lparams:     the params to be given to the learner (Excludes data and weightidx, the first two terms in the learner params)
# T:           the number of iterations to run
# giveErrors:  True if the classifier should be tested at each iteration, and it's errors reported.
# testdata:    Testing data for calculating test error.
# Returns:     Returns an AdaBoosted classifier.
#              If giveErrors=True, also returns a list of T rows containing the following for each iteration:
#               Final classifier training error
#               Final classifier testing error (None if no testing data is given)
#               weak classifier t training error
#               weak classifier t testing error (None if no testing data is given)
def adaboost(data, labelidx, learner, lparams, T, giveErrors=False, testdata=None):
	# Initialize the values
	w = 1 / len(data)
	weightidx = AddColumn(data, w)
	adaclassif = Adaboost([], [])
	if giveErrors:
		errs = []
	
	# Go through the iterations
	for i in range(T):
		# Generate a weak classifier
		curclassif = learner(data, weightidx, *lparams)
		adaclassif.classifs.append(curclassif)
		# Calculate error
		wtrainerr = CalculateError(data, labelidx, curclassif, weightidx)
		cura = 0.5 * math.log((1-wtrainerr) / wtrainerr)
		adaclassif.a.append(cura)
		# Calculate other errors
        # These reported errors are *unweighted*
		if giveErrors:
			curerrs = []
			curerrs.append(CalculateError(data, labelidx, adaclassif, None))                                 # ftrainerr
			curerrs.append(None if testdata==None else CalculateError(testdata, labelidx, adaclassif, None)) # ftesterr
			curerrs.append(CalculateError(data, labelidx, curclassif, None))                                 # wtrainerr
			curerrs.append(None if testdata==None else CalculateError(testdata, labelidx, curclassif, None)) # wtesterr
			errs.append(curerrs)
				
		# Update the weights
		total = 0
		for d in data:
			update = d[weightidx] * math.exp(-cura * (1 if d[labelidx] == curclassif.PredictLabel(d) else -1))
			d[weightidx] = update
			total += update
		for d in data:
			d[weightidx] /= total
			
	# Return the full classifier, and errors (if requested)
	if not giveErrors:
		return adaclassif
	else:
		return (adaclassif, errs)
			
		
  
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			