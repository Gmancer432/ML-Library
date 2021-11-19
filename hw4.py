
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
        (r_0, a) = args # current best: (1, 1/10)
        return r_0/(1+(r_0*t)/a)
    
    T = 100
    # These values are in the following format:
    # (C, startri, endri, startai, endai)
    vals = []
    vals.append((100/873, 1/10, 1/10))
    #vals.append((500/873, -5, 5, -5, 5))
    #vals.append((700/873, -20, 20, -100, 0))
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
        print('Loss graph: (opens a window, close window to continue)')
        x = np.linspace(0, len(losses), num=len(losses))
        plt.plot(x, losses)
        plt.show()
        print()
    
else:
	print('Skipped!\n')
    
    
    
## Part 2.b ##
# Run SVM for 100 epochs with r_t = r_0/(1+t)
# Report the final weights and the average error over the test data

print('Part 2.b\n')

# Set to True to run this part

if False:
    
    def rfunc(t, r_0):
        return r_0/(1+t)
    
    T = 100
    # These values are in the following format:
    # (C, startri, endri, startai, endai)
    vals = []
    vals.append((100/873, -5, 5, -5, 5))
    #vals.append((500/873, -5, 5, -5, 5))
    #vals.append((700/873, -20, 20, -100, 0))
    for (C, startri, endri, startai, endai) in vals:
        # Tune the hyperparameters
        bestr0 = None
        besta = None
        bestmodel = None
        bestlosses = None
        bestscore = None
        for ri in np.linspace(startri, endri, num=11):
            for ai in np.linspace(startai, endai, num=11):
                r_0 = 10**ri
                a = 10**ai
                w = np.zeros(trdata.shape[1])
                model = SVM(w)
                losses = SVMSGD(trdata, trlabels, model, T, C, rfunc, r_0, ReportObjFunc=True)
                #score = abs(np.average(losses[-1000]) - losses[-1])
                score = AverageError(trdata, trlabels, model)
                if bestscore == None or score < bestscore:
                    bestr0 = r_0
                    besta = a
                    bestmodel = model
                    bestlosses = losses
                    bestscore = score
        
        # Report the results
        print('Results for C=' + str(C) + ':')
        # Found hyperparameters
        print('Tuned hyperparameters:')
        print('\t r_0 = ' + str(bestr0))
        print('\t a = ' + str(besta))
        # Final weights
        print('Final weights [b, w_0, w_1, ...]:')
        print(bestmodel.w)
        # Final Errors
        print('Training error: ' + str(AverageError(trdata, trlabels, bestmodel)))
        print('Testing error: ' + str(AverageError(tstdata, tstlabels, bestmodel)))
        # Loss graph:
        print('Loss graph: (opens a window, close window to continue)')
        x = np.linspace(0, len(bestlosses), num=len(bestlosses))
        plt.plot(x, bestlosses)
        plt.show()
        print()
    
else:
	print('Skipped!\n')
    
    
    
    
    
    