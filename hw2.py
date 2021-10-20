
from Util.util import *
from DecisionTree.ID3 import *
from EnsembleLearning.adaboost import *

### temp import!
import winsound

### Part 2.b ###
    
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


### temp cmd!!
for i in (300, 500, 400):
    winsound.Beep(i, 1000)










