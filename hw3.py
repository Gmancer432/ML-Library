
from Util.util import ReadCSV, ConvertColumn, PrintCSV
from Perceptron.perceptron import *


# Import the raw data
rawtraining = ReadCSV('Data/bank-note data/train.csv')
rawtesting = ReadCSV('Data/bank-note data/test.csv')

# Convert columns to their appropriate data types
for c in (0, 1, 2, 3):
    for d in (rawtraining, rawtesting):
        ConvertColumn(d, c, float)
for d in (rawtraining, rawtesting):
    ConvertColumn(d, 4, int)

# Separate the data from the labels and put them into np arrays
# Also convert labels to {-1, 1}, and augment the data with [1, x_1, x_2, ...]
trdata = np.array([[1, *d[:4]] for d in rawtraining])
trlabels = np.array([1 if d[4] == 1 else -1 for d in rawtraining])
tstdata = np.array([[1, *d[:4]] for d in rawtesting])
tstlabels = np.array([1 if d[4] == 1 else -1 for d in rawtesting])



## Part 2.a ##
# Run standard Percecptron for 10 epochs
# Report the final weights and the average error over the test data

print('Part 2.a\n')

# Set to True to run this part

if True:
    # Create a standard Perceptron model
    w = np.zeros(trdata.shape[1])
    r = 1  # the learning rate isn't given in the homework, so this is assumed
    T = 10
    pmodel = PerceptronStandard(w, r)
    # Run the perceptron algorithm
    RunPerceptron(trdata, trlabels, pmodel, T)
    # Report the results
    print('Learned Weights (b, w_1, w_2, ...):')
    print(pmodel.w)
    print('Average Error: ' + str(AverageError(tstdata, tstlabels, pmodel)))
    print()
else:
	print('Skipped!\n')
    
    
    
## Part 2.b ##
# Run voted Percecptron for 10 epochs
# Report the final weights and the average error over the test data

print('Part 2.b\n')

# Set to True to run this part

if True:
    # Create a standard Perceptron model
    w = np.zeros(trdata.shape[1])
    r = 1  # the learning rate isn't given in the homework, so this is assumed
    T = 10
    pmodel = PerceptronVoted(w, r)
    # Run the perceptron algorithm
    RunPerceptron(trdata, trlabels, pmodel, T)
    # Report the results
    print('idx, count, Learned Weights (b, w_1, w_2, ...):\n')
    warray = [[i, pmodel.wlist[i][1], *pmodel.wlist[i][0]] for i in range(len(pmodel.wlist))]
    PrintCSV(warray)
    print('Average Error: ' + str(AverageError(tstdata, tstlabels, pmodel)))
    print()
else:
	print('Skipped!\n')



## Part 2.c ##
# Run averaged Percecptron for 10 epochs
# Report the final weights and the average error over the test data

print('Part 2.c\n')

# Set to True to run this part

if True:
    # Create a standard Perceptron model
    w = np.zeros(trdata.shape[1])
    r = 1  # the learning rate isn't given in the homework, so this is assumed
    T = 10
    pmodel = PerceptronAveraged(w, r)
    # Run the perceptron algorithm
    RunPerceptron(trdata, trlabels, pmodel, T)
    # Report the results
    print('Learned Weights (b, w_1, w_2, ...):')
    print(pmodel.a)
    print('Average Error: ' + str(AverageError(tstdata, tstlabels, pmodel)))
    print()
else:
	print('Skipped!\n')






































