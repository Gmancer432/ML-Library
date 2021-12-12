
from Util.util import ReadCSV, ConvertColumn, PrintCSV
from NeuralNetworks.nn import *
import matplotlib.pyplot as plt


# Import the raw data
rawtraining = ReadCSV('Data/bank-note data/train.csv')
rawtesting = ReadCSV('Data/bank-note data/test.csv')

# Convert columns to their appropriate data types
for c in (0, 1, 2, 3):
    for d in (rawtraining, rawtesting):
        ConvertColumn(d, c, float)
for d in (rawtraining, rawtesting):
    ConvertColumn(d, 4, int)

# Put the data into np arrays
trdata = np.array([[*d[:4]] for d in rawtraining])
trlabels = np.array([1 if d[4] == 1 else -1 for d in rawtraining])
tstdata = np.array([[*d[:4]] for d in rawtesting])
tstlabels = np.array([1 if d[4] == 1 else -1 for d in rawtesting])
(trnum, dim) = trdata.shape



print('test\n')

# Set to True to run this part

if True:

    testdata = np.array([1, 2, 3, 4])
    model = ClassifNN(4, 2, initones)
    model.w_0 /= 2
    model.w_1 /= 3
    model.w_f /= 4
    model.gradientdescentstep(testdata, 1, 0.1)
    
    
else:
	print('Skipped!\n')
    
    
    














