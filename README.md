# ML-Library
This is a machine learning library developed by Sean Richens for CS5350/6350 in University of Utah.

-- Decision Tree documentation --

First, read in the data.
Util.ReadCSV(filepath) -> list of rows of data
If the datatype of a column needs to be convered, use 
Util.ConvertColumn(data, columnidx, datatype)

Before training a decision tree, you need to prepare a dictionary of attributes:
{ 'name' : ID3.Attribute(...), ... }
An attribute object describes information the tree needs to know about the attribute:
ID3.Attribute(name, idx, values, numval=False):
- name:    the string name of the attribute
- idx:     the index of the attribute in a row of data
- values:  a list of values the attribute can take, if categorical.  If the attribute is numerical, this list should be empty.
- numval:  defaults to False.  Set to True if the attribute is numerical.

Currently, if you have data with missing values, you need to process that data yourself.

From here, a decision tree can be trained:
	ID3.ID3(data, atrsDict, atrsList, purityfn, maxdepth)
		- data:  a list of rows of data, probably taken from Util.ReadCSV(..)
		- atrsDict:  A prepared dictionary of attributes
		- atrsList:  A list of string names of the attributes
		- purityfn:  The chosen purity function.  Choose from ID3.Entropy, ID3.ME, or ID3.GI
		- maxdepth:  The maximum depth of the tree.  0=root only, None=no limit
		- maxatrs:  Used to create a Random Tree (usually as part of a Random Forest. Defaults to None, which means no limit.  If maxatrs is an int > 0, each node will limit the number of possible attributes it can pick from to the value of maxatrs.

Finally, you can calculate error with CalculateError(..)
	CalculateError(data, labelidx, DT)
		- data:  List of rows of data
		- labelidx:  index of the label for the row.  Usually the last item in the row.
		- DT:  The decision tree to calculate error for.


-- Adaboost documentation --

The Adaboost implementation is most compatible with Decision Trees.  Each row of training data should include a column for its label, but should not have a column for weight (this is added at the start of the algorithm).

Run adaboost with the following parameters:
adaboost(data, labelidx, learner, lparams, T, giveErrors, testdata)
- data:  the input data, including a column for label but excluding a column for weight
- labelidx:  the index of the weight column
- learner:  a pointer to the learner to use (Ex: ID3)
- lparams:  a tuple of additional params to give to the learner.  This excludes the data and weightidx, which are assumed to be the first two parameters of the learner.
- T:  the number of iterations to run
- giveErrors:  defaults to False. True if, after each iteration, the current weak classifier and the final classifier (so far) should be tested.  Returns a list of training errors for both.  If test data is provided, also returns a list of test errors for both.
- testdata: defaults to none.  If giveErrors=True, also returns errors for test data.
- returns:  (classifier, errors)
	errors is a list of four errors for each iteration:  
	- final training error
	- final testing error (or None if no testing data)
	- weak training error
	- weak testing error (or None if no testing data)
Use classif.PredictLabel(datapoint) to get a prediction.


-- Bagging documentation --

The Bagging implementation is most compatible with Decision Trees.  Each row of training data should include a column for its label, but should not have a column for weight (this is added at the start of the algorithm).

Run bagging with the following parameters:
bagging(data, labelidx, learner, lparams, T, m, giveFinalErrors, giveSingleErrors, testdata)
- data:  the input data, including a column for label but excluding a column for weight
- labelidx:  the index of the weight column
- learner:  a pointer to the learner to use (Ex: ID3)
- lparams:  a tuple of additional params to give to the learner.  This excludes the data and weightidx, which are assumed to be the first two parameters of the learner.
- T:  the number of iterations to run
- giveFinalErrors:  defaults to False. True if, after each iteration, the final classifier (so far) should be tested.  Returns a list of training errors.  If test data is provided, also returns a list of test.
- giveSingleErrors:  defaults to False. True if, after each iteration, the current single classifier should be tested.  Returns a list of training errors.  If test data is provided, also returns a list of test.
- testdata: defaults to none.  If giveFinalErrors=True or giveSingleErrors=True, also returns errors for test data for the appropriate classifier(s).
- returns:  (classifier, errors)
	errors is a list of (up to) four errors for each iteration, depending on what is asked for:  
	- final training error
	- final testing error
	- single classif training error
	- single classif testing error	
Use classif.PredictLabel(datapoint) to get a prediction.


-- Random Forest Documentation --

A Random Forest is just Bagged Decision Trees, where each decision tree has maxatrs != None.  

maxatrs is a parameter of the ID3 function that defaults to None.  Change this parameter to some int > 0 to create Random Trees, and bag these trees to create a Random Forest.


-- LMS Documentation --

The LMS class can be found in LinearRegression.Linear.

The LMS class is created with a numpy array of weights (w).  If you want outputs to be constrained to {1, -1}, set classifier=True.  
Data, labels, and weights that are used in LMS must be in their own numpy arrays.  Also keep in mind that data=(1, x_1, x_2, ...) and weights=(b, w_1, w_2, ...)

LMS.PredictLabel(data) will output labels for a set of data.  Data must be a numpy array, but any number of datapoints may be given.

LMS.Loss(data, labels) will output loss for a set of given data.

LMS.Gradient(data, labels) will return a gradient wrt the weights.  This is used in Gradient Descent.

AnalyticalLMS(data, labels) will return the optimal weights according to the LMS loss function, computed analytically.


-- Gradient Descent Documentation --

The Gradient Descent algorithm can be found in LinearRegression.GradientDescent.

The Gradient Descent algorithm should be able to take in any classifer model that has a classifier.Gradient(data, labels) function.  Call GradientDescent with the following parameters:

GradientDescent(data, labels, classifier, batchsize, r, tolerance=None, T=None, giveTrainingLoss=False, giveTestingLoss=False, testingdata=None, testinglabels=None)
- data:  a numpy array of data.  Unlike with the decision trees, this should not have any labels
- labels: a numpy array of labels.  label[i] must correspond to data[i].
- classifier:  a classifier object.  This object must have a classifier.Gradient(data, labels) function, which returns an appropriate gradient.
- batchsize:  the amount of data to perform the gradient over at each iteration.  batchsize=1 => stochastic gradient descent.
- r: The learning rate; a float in the range of (0, 1)
- tolerance:  defaults to None.  Currently designed for LMS, don't use if your classifier doesn't have classif.w (weights).  If the change in w is less than this tolerance, then learning stops.
- T: The number of iterations to run.  T=None => no limit. !!Make sure to have T!= None if you set tolerance=None!!  It is recommended that you set T!=None regardless of what tolerance is set to.
- giveTrainingLoss:  defaults to False. If True, finds the training loss at each iteration (over the current batch of data) and returns the list at the end.
- giveTestingLoss:  defaults to False. If True, finds the testing loss (over all the testing data) at each iteration and returns the list at the end.
- testingdata:  the testing data to test on if giveTestingLoss=True
- testinglabels:  the testing labels to test on if giveTestingLoss=True
- return:  A tuple of (up to) three items:
	- classifier
	- training losses
	- testing losses















