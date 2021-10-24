
from Util.util import *
from DecisionTree.ID3 import *
from EnsembleLearning.adaboost import adaboost
from EnsembleLearning.bagging import bagging, BiasVariance
from LinearRegression.Linear import *
from LinearRegression.GradientDescent import *
import random
import numpy as np
import matplotlib.pyplot as plt

### temp import!
import winsound

### Part 2.b ###
print('Part 2.a\n')
	
# Retrieve the bank data
trainingdata = ReadCSV('Data/bank data/train.csv')
testingdata = ReadCSV('Data/bank data/test.csv')

# Prepare the Attributes
atrs = {
	'age'       : Attribute('age',       0,  [], True),
	'job'       : Attribute('job',       1,  ['admin.', 'unknown', 'unemployed', 'management', 
											  'housemaid', 'entrepreneur', 'student', 'blue-collar', 
											  'self-employed', 'retired', 'technician', 'services']), #unk
	'marital'   : Attribute('marital',   2,  ['married', 'divorced', 'single']),
	'education' : Attribute('education', 3,  ['unknown', 'secondary', 'primary', 'tertiary']),        #unk
	'default'   : Attribute('default',	 4,  ['yes', 'no']),
	'balance'   : Attribute('balance',   5,  [], True),
	'housing'   : Attribute('housing',   6,  ['yes', 'no']),
	'loan'      : Attribute('loan',      7,  ['yes', 'no']),
	'contact'   : Attribute('contact',   8,  ['unknown', 'telephone', 'cellular']),                   #unk
	'day'       : Attribute('day',       9,  [], True),
	'month'     : Attribute('month',     10, ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
											  'sep', 'oct', 'nov', 'dec']),
	'duration'  : Attribute('duration',  11, [], True),
	'campaign'  : Attribute('campaign',  12, [], True),
	'pdays'     : Attribute('pdays',     13, [], True),
	'previous'  : Attribute('previous',  14, [], True),
	'poutcome'  : Attribute('poutcome',  15, ['unknown', 'other', 'failure', 'success']),              #unk
	'label'     : Attribute('label',     16, ['yes', 'no'])	
}

# Convert numerical categories into numbers
for i in (0, 5, 9, 11, 12, 13, 14):
	ConvertColumn(trainingdata, i, int)
	ConvertColumn(testingdata, i, int)

# Here is where the boosting starts
# set to True to perform the learning

if True:
	# Prepare the learning parameters
	lparams = (atrs, 
			   ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
				'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 
				'previous', 'poutcome'],  
			   Entropy, 
			   1)

	(adaclassif, errs) = adaboost(trainingdata.copy(), 16, ID3, lparams, 500, True, testingdata)

	# print the errors
	print('Errors:')
	print('Rows: iterations 1-500')
	print('Columns: Final training, Final testing, Stump t training, Stump t testing')
	print()
	PrintCSV(errs)
else:
	print('Skipped!\n')



### Part 2.b
print('Part 2.b\n')

# Here is where the bagging starts
# set to True to perform the learning

if True:
	lparams = (atrs, 
			   ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
				'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 
				'previous', 'poutcome'],  
			   Entropy, 
			   None)

	# the value for m' isn't given, so I chose 1000
	(baggedclassif, errs) = bagging(trainingdata.copy(), 16, ID3, lparams, 500, 1000, giveFinalErrors=True, giveSingleErrors=True, testdata=testingdata)

	# print the errors
	print('Errors:')
	print('Rows: iterations 1-500')
	print('Columns: Final training, Final testing, Tree t training, Tree t testing')
	print()
	PrintCSV(errs)
else:
	print('Skipped!\n')





### Part 2.c
print('Part 2.c\n')

lparams = (atrs, 
		   ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
			'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 
			'previous', 'poutcome'],  
		   Entropy, 
		   None)
	  
# Here is where the bagging starts
# set to True to perform the learning

if True:
	# Collect 100 bagged classifiers
	baggedclassifs = []
	totalbias = 0
	totalsvar = 0
	for i in range(100):
		# Sample 1000 examples uniformly *without* placement
		data = random.sample(trainingdata, 1000)
		# Collect a bag of 500 trees
		baggedclassifs.append(bagging(CopyData(data), 16, ID3, lparams, 500, 1000))
	# Collect the average bias and variance of the single trees
	singletrees = []
	for b in baggedclassifs:
		singletrees.append(b.classifs[0])
	(sbias, svar) = BiasVariance(testingdata, 16, singletrees, 'yes')
	print('Single Trees:')
	print('Bias:  ' + str(sbias))
	print('Variance:  ' + str(svar))
	print('General squared error:  ' + str(sbias + svar))
	print()
	# Collet the average bias and variance of the bagged trees
	(bbias, bvar) = BiasVariance(testingdata, 16, baggedclassifs, 'yes')
	print('Bagged Trees:')
	print('Bias:  ' + str(bbias))
	print('Variance:  ' + str(bvar))
	print('General squared error:  ' + str(bbias + bvar))
	print()
		
else:
	print('Skipped!\n')



### Part 2.d
print('Part 2.d\n')

# the RandomForest learner just uses the Bagging algorithm, 
# except lparams has an extra parameter (maxatrs={2, 4, 6})
maxatrs = 2
lparams = (atrs, 
			   ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
				'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 
				'previous', 'poutcome'],  
			   Entropy, 
			   None,
			   maxatrs)

# set to True to perform the learning

if True:
	# prepare error list
	errlist = []
	for i in range(500):
		errlist.append([])
	# train random forests
	for a in (2, 4, 6):
		maxatrs = a
		(baggedclassif, errs) = bagging(trainingdata.copy(), 16, ID3, lparams, 500, 1000, giveFinalErrors=True, giveSingleErrors=False, testdata=testingdata)
		# fit the data rows into the full errlist
		for i in range(len(errlist)):
			errlist[i] += errs[i]

	# print the errors
	print('Errors:')
	print('Rows: iterations 1-500')
	print('Columns: Final training and Final testing, for max atrs = {2, 4, 6}')
	print()
	PrintCSV(errlist)
	print()
		
else:
	print('Skipped!\n')



### Part 2.e
print('Part 2.e\n')

maxatrs = 2
lparams = (atrs, 
			   ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
				'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 
				'previous', 'poutcome'],  
			   Entropy, 
			   None,
			   maxatrs)

# set to True to perform the learning

if True:
	# Collect 100 random forests
	baggedclassifs = []
	totalbias = 0
	totalsvar = 0
	for i in range(100):
		# Sample 1000 examples uniformly *without* placement
		data = random.sample(trainingdata, 1000)
		# Collect a bag of 500 trees
		baggedclassifs.append(bagging(CopyData(data), 16, ID3, lparams, 500, 1000))
	# Collect the average bias and variance of the single trees
	singletrees = []
	for b in baggedclassifs:
		singletrees.append(b.classifs[0])
	(sbias, svar) = BiasVariance(testingdata, 16, singletrees, 'yes')
	print('Single Trees:')
	print('Bias:  ' + str(sbias))
	print('Variance:  ' + str(svar))
	print('General squared error:  ' + str(sbias + svar))
	print()
	# Collet the average bias and variance of the bagged trees
	(bbias, bvar) = BiasVariance(testingdata, 16, baggedclassifs, 'yes')
	print('Bagged Trees:')
	print('Bias:  ' + str(bbias))
	print('Variance:  ' + str(bvar))
	print('General squared error:  ' + str(bbias + bvar))
	print()
		
else:
	print('Skipped!\n')




## Part 4.a
print('Part 4.a\n')

# collect and process the concrete data
# these values need to be in numpy arrays, and the labels separate from the data
rawtrainingdata = ReadCSV('Data/concrete data/train.csv')
rawtestingdata = ReadCSV('Data/concrete data/test.csv')

# Convert all data into numbers
for i in range(len(rawtrainingdata[0])):
	ConvertColumn(rawtrainingdata, i, float)
	ConvertColumn(rawtestingdata, i, float)

trdata = np.array([[1, *d[0:7]] for d in rawtrainingdata])
trlabels = np.array([d[7] for d in rawtrainingdata])
tstdata = np.array([[1, *d[0:7]] for d in rawtestingdata])
tstlabels = np.array([d[7] for d in rawtestingdata])

# set to True to perform the learning

if True:
	# initialize the weight vector and bias
	w = np.ones(8)
	classif = LMS(w)
	# Perform gradient descent
	lr = 0.01
	(classif, losses) = GradientDescent(trdata, trlabels, classif, trdata.shape[0], lr, tolerance=1e-6, T=15, giveTrainingLoss=True)
	# print the losses
	print('Weight vector [b, w_1, w_2, ...]:')
	print(classif.w)
	print('Learning Rate:  ' + str(lr))
	print('Final Testing Loss:  ' + str(classif.Loss(tstdata, tstlabels)))
	print('Training Loss at each step: (opens a window, close window to continue)')
	x = np.linspace(0, len(losses), num=len(losses))
	plt.plot(x, losses)
	plt.show()
	
		
else:
	print('Skipped!\n')



## Part 4.b
print('Part 4.b\n')

# set to True to perform the learning

if True:
	# initialize the weight vector and bias
	w = np.ones(8)
	classif = LMS(w)
	# Perform gradient descent
	lr = 0.001
	(classif, losses) = GradientDescent(trdata, trlabels, classif, 1, lr, tolerance=None, T=7000, 
                                        giveTestingLoss=True, testingdata=tstdata, testinglabels=tstlabels)
	# print the losses
	print('Weight vector [b, w_1, w_2, ...]:')
	print(classif.w)
	print('Learning Rate:  ' + str(lr))
	print('Final Testing Loss:  ' + str(classif.Loss(tstdata, tstlabels)))
	print('Testing Loss at each step: (opens a window, close window to continue)')
	x = np.linspace(0, len(losses), num=len(losses))
	plt.plot(x, losses)
	plt.show()
	
		
else:
	print('Skipped!\n')


## Part 4.c
print('Part 4.c\n')


if True:
	# Here, w* is calculated analytically with a formula
	# Remember that w[0] = b
	w = AnalyticalLMS(trdata, trlabels)
	print(w)
		
else:
	print('Skipped!\n')



### temp cmd!!
for i in (300, 500, 400):
	winsound.Beep(i, 1000)










