
# Analysis for homework 1


from util import *
from ID3 import *


### Part 2 ###

print('Question 2:\n')

# Retrieve the car data
trainingdata = ReadCSV('car data/train.csv')
testingdata = ReadCSV('car data/test.csv')

# Prepare the attributes
atrs = {
	'buying'   : Attribute('buying',   0, ['vhigh', 'high', 'med', 'low']),
	'maint'    : Attribute('maint',    1, ['vhigh', 'high', 'med', 'low']),
	'doors'    : Attribute('doors',    2, ['2', '3', '4', '5more']),
	'persons'  : Attribute('persons',  3, ['2', '4', 'more']),
	'lug_boot' : Attribute('lug_boot', 4, ['small', 'med', 'big']),
	'safety'   : Attribute('safety',      5, ['low', 'med', 'high']),
	'label'    : Attribute('label',    6, ['unacc', 'acc', 'good', 'vgood'])
}

# Calculate error for each heuristic and depth
trainingerrors = []
testingerrors = []
for depth in range(1, 7):
	curtrainingerrors = []
	curtestingerrors = []
	for h in (Entropy, ME, GI):
		DT = ID3(trainingdata, 
				 atrs, 
				 ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'], 
				 h, 
				 depth)
		curtrainerror = CalculateError(trainingdata, 6, DT)
		curtesterror = CalculateError(testingdata, 6, DT)
		curtrainingerrors.append(curtrainerror)
		curtestingerrors.append(curtesterror)
	trainingerrors.append(curtrainingerrors)
	testingerrors.append(curtestingerrors)

# Print the table
print('Training Errors:')
print('Rows: depth 1-6')
print('Columns: Entropy, ME, GI')
print()
PrintCSV(trainingerrors)
print()
print('Testing Errors:')
print('Rows: depth 1-6')
print('Columns: Entropy, ME, GI')
print()
PrintCSV(testingerrors)
print('\n')



### Part 3 ###

# Retrieve the bank data
trainingdata = ReadCSV('bank data/train.csv')
testingdata = ReadCSV('bank data/test.csv')

# Convert numerical categories into numbers
for i in (0, 5, 9, 11, 12, 13, 14):
	ConvertColumn(trainingdata, i, int)
	ConvertColumn(testingdata, i, int)
	
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

## part a
print('Question 3, part a\n')

# Calculate error for each heuristic and depth
trainingerrors = []
testingerrors = []
for depth in range(1, 17):
	curtrainingerrors = []
	curtestingerrors = []
	for h in (Entropy, ME, GI):
		DT = ID3(trainingdata, 
				 atrs, 
				 ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
				  'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 
				  'previous', 'poutcome'],  
				 h, 
				 depth)
		curtrainerror = CalculateError(trainingdata, 16, DT)
		curtesterror = CalculateError(testingdata, 16, DT)
		curtrainingerrors.append(curtrainerror)
		curtestingerrors.append(curtesterror)
	trainingerrors.append(curtrainingerrors)
	testingerrors.append(curtestingerrors)

# Print the table
print('Training Errors:')
print('Rows: depth 1-16')
print('Columns: Entropy, ME, GI')
print()
PrintCSV(trainingerrors)
print()
print('Testing Errors:')
print('Rows: depth 1-16')
print('Columns: Entropy, ME, GI')
print()
PrintCSV(testingerrors)
print('\n')


## part b
print('Question 3, part b\n')

# for each category with unkowns in it, find the majority value and replace the unkowns with this value
for unkcat in ('job', 'education', 'contact', 'poutcome'):
	# Count the values
	labelCounts, totalCount = CountAttributeCat(trainingdata, unkcat, atrs)
	# Zero out the unknown value count
	valuelist = atrs[unkcat].values
	unkidx = valuelist.index('unknown')
	labelCounts[unkidx] = 0
	# Find the majority value
	maxCount = max(labelCounts)
	majorityidx = labelCounts.index(maxCount)
	majoritylabel = valuelist[majorityidx]
	# replace unk with the majority value
	idx = atrs[unkcat].idx
	for t in (trainingdata, testingdata):
		for d in t:
			if d[idx] == 'unknown':
				d[idx] = majoritylabel
	# update the atrsDict
	valuelist.pop(unkidx)

# Do the same testing as before

# Calculate error for each heuristic and depth
trainingerrors = []
testingerrors = []
for depth in range(1, 17):
	curtrainingerrors = []
	curtestingerrors = []
	for h in (Entropy, ME, GI):
		DT = ID3(trainingdata, 
				 atrs, 
				 ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
				  'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 
				  'previous', 'poutcome'],  
				 h, 
				 depth)
		curtrainerror = CalculateError(trainingdata, 16, DT)
		curtesterror = CalculateError(testingdata, 16, DT)
		curtrainingerrors.append(curtrainerror)
		curtestingerrors.append(curtesterror)
	trainingerrors.append(curtrainingerrors)
	testingerrors.append(curtestingerrors)

# Print the table
print('Training Errors:')
print('Rows: depth 1-16')
print('Columns: Entropy, ME, GI')
print()
PrintCSV(trainingerrors)
print()
print('Testing Errors:')
print('Rows: depth 1-16')
print('Columns: Entropy, ME, GI')
print()
PrintCSV(testingerrors)
print('\n')









  