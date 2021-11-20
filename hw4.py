
from Util.util import ReadCSV, ConvertColumn, PrintCSV
from SVM.SVM import *
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

# Separate the data from the labels and put them into np arrays
# Also convert labels to {-1, 1}, and augment the data with [1, x_1, x_2, ...]
trdata = np.array([[1, *d[:4]] for d in rawtraining])
trlabels = np.array([1 if d[4] == 1 else -1 for d in rawtraining])
tstdata = np.array([[1, *d[:4]] for d in rawtesting])
tstlabels = np.array([1 if d[4] == 1 else -1 for d in rawtesting])



## Part 2.a ##
# Run SVM for 100 epochs with r_t = r_0/(1+(r_0*t)/a)
# Report the final weights and the average error over the test data

print('Part 2.a\n')

# Set to True to run this part

if True:
    
    def rfunc(t, args):
        (r_0, a) = args
        return r_0/(1+(r_0*t)/a)
    
    T = 100
    # These values are in the following format:
    # (C, r_0, a)
    vals = []
    vals.append((100/873, 1, 1/100))
    vals.append((500/873, 1/10, 1/100))
    vals.append((700/873, 1/100, 1/100))
    for (C, r_0, a) in vals:
        w = np.zeros(trdata.shape[1])
        model = SVM(w)
        losses = SVMSGD(trdata, trlabels, model, T, C, rfunc, (r_0, a), ReportObjFunc=True)
                
        # Report the results
        print('Results for C=' + str(C) + ':')
        # Found hyperparameters
        print('Tuned hyperparameters:')
        print('\t r_0 = ' + str(r_0))
        print('\t a = ' + str(a))
        # Final weights
        print('Final weights [b, w_0, w_1, ...]:')
        print(model.w)
        # Final Errors
        print('Training error: ' + str(AverageError(trdata, trlabels, model)))
        print('Testing error: ' + str(AverageError(tstdata, tstlabels, model)))
        # Loss graph:
        #print('Loss graph: (opens a window, close window to continue)')
        x = np.linspace(0, len(losses), num=len(losses))
        plt.plot(x, losses)
        #plt.show()
        print()
    
else:
	print('Skipped!\n')
    
    
    
## Part 2.b ##
# Run SVM for 100 epochs with r_t = r_0/(1+t)
# Report the final weights and the average error over the test data

print('Part 2.b\n')

# Set to True to run this part

if True:
    
    def rfunc(t, r_0):
        return r_0/(1+t)
    
    T = 100
    # These values are in the following format:
    # (C, r_0)
    vals = []
    vals.append((100/873, 1/10))
    vals.append((500/873, 1/100))
    vals.append((700/873, 1/100))
    for (C, r_0) in vals:
        w = np.zeros(trdata.shape[1])
        model = SVM(w)
        losses = SVMSGD(trdata, trlabels, model, T, C, rfunc, r_0, ReportObjFunc=True)
                
        # Report the results
        print('Results for C=' + str(C) + ':')
        # Found hyperparameters
        print('Tuned hyperparameters:')
        print('\t r_0 = ' + str(r_0))
        # Final weights
        print('Final weights [b, w_0, w_1, ...]:')
        print(model.w)
        # Final Errors
        print('Training error: ' + str(AverageError(trdata, trlabels, model)))
        print('Testing error: ' + str(AverageError(tstdata, tstlabels, model)))
        # Loss graph:
        #print('Loss graph: (opens a window, close window to continue)')
        x = np.linspace(0, len(losses), num=len(losses))
        plt.plot(x, losses)
        #plt.show()
        print()
    
else:
	print('Skipped!\n')


# the data for the Dual SVM is *not* augmented
trdata = np.array([[*d[:4]] for d in rawtraining])
trlabels = np.array([1 if d[4] == 1 else -1 for d in rawtraining])
tstdata = np.array([[*d[:4]] for d in rawtesting])
tstlabels = np.array([1 if d[4] == 1 else -1 for d in rawtesting])

## Part 3.a ##
# Run dual SVM
# Report the final weights and the average error over the test data

print('Part 3.a\n')

# Set to True to run this part

if True:

    vals = []
    vals.append(100/873)
    vals.append(500/873)
    vals.append(700/873)
    for C in vals:
        model = SVMDual(kernel=LinearKernel)
        model.Optimize(trdata, trlabels, C)
                
        # Report the results
        print('Results for C=' + str(C) + ':')
        # Final weights
        wstar = (model.svdata.T * model.svlabels * model.svastar).T.sum(axis=0)
        print('Final weights [w_0, w_1, ...]:')
        print(wstar)
        print('Final bias (b): ' + str(model.b))
        # Final Errors
        print('Training error: ' + str(AverageError(trdata, trlabels, model)))
        print('Testing error: ' + str(AverageError(tstdata, tstlabels, model)))
        print()
    
else:
	print('Skipped!\n')




## Part 3.b ##
# Run Dual SVM with the Gaussian kernel
# Report the average error over the test data, and the number of support vectors for each combination of C, gamma
# for C=500/873, report how many support vectors overlap between consecutive values of gamma

print('Part 3.b/c\n')

# Set to True to run this part

if True:

    cvals = []
    cvals.append(100/873)
    cvals.append(500/873)
    cvals.append(700/873)
    gammavals = []
    gammavals.append(0.1)
    gammavals.append(0.5)
    gammavals.append(1)
    gammavals.append(5)
    gammavals.append(100)
    prevsvs = None
    for C in cvals:
        for g in gammavals:
            model = SVMDual(kernel=GaussianKernel, kernelargs=g)
            model.Optimize(trdata, trlabels, C)
                    
            # Report the results
            print('Results for C=' + str(C) + ', gamma=' + str(g) + ':')
            # Final Errors
            print('Training error: ' + str(AverageError(trdata, trlabels, model)))
            print('Testing error: ' + str(AverageError(tstdata, tstlabels, model)))
            # Number of support vectors
            print('Number of support vectors: ' + str(model.svdata.shape[0]))
            # for C=500/873, find the consecutive values
            if C == 500/873:
                if prevsvs is not None:
                    dists = scipy.spatial.distance.cdist(prevsvs, model.svdata, 'sqeuclidean')
                    numoverlapping = dists[dists<model.svlim].size
                    print('There are [' + str(numoverlapping) + '] overlapping support vectors from the previous value of gamma')
                prevsvs = model.svdata
            print()
    
else:
	print('Skipped!\n')
    
    
    
    
    
    