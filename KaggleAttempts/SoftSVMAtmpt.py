
from Util.util import ConvertColumn, SaveCSV
from SVM.SVM import *
import numpy as np
import matplotlib.pyplot as plt
import math

### Runs the data through Linear Regression

def main(raw_training, raw_testing):
    # Prepare the data
    # Categorical attributes are converted into vectors of boolean attributes
    
    # Convert the columns that are already numbers
    for t in (raw_training, raw_testing):
        for c in (0, 2, 4, 10, 11, 12):
            ConvertColumn(t, c, float)
    ConvertColumn(raw_training, 14, int)
    
    # Preprocessing data and functions

    atrsDictCat = {
    # workclass               
                    1  : ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
    # education      
                    3  : ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
    # marital-status
                    5  : ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
    # occupation                
                    6  : ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
    # relationship     
                    7  : ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    # race               
                    8  : ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    # sex           
                    9  : ['Female', 'Male'],
    # native-country    
                    13 : ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],
                    }

    # Converts string values to vector embeddings
    vectDict = {}
    for i in atrsDictCat.keys():
      valueList = atrsDictCat[i]
      vocablen = len(valueList)
      vectorlen = math.ceil(math.log(vocablen, 2))
      for val in valueList:
          vectDict[val] = np.random.normal(size=vectorlen) / vectorlen**0.5

    # Call this method for '?' values
    def get0Embedding(index):
      valueList = atrsDictCat[index]
      vocablen = len(valueList)
      vectorlen = math.ceil(math.log(vocablen, 2))
      return np.zeros(vectorlen)
    
    # convert to vectorized nparray
    labeled_data = []
    for r in raw_training:
      l = [1]
      for cidx in range(14):
        if cidx in (0, 2, 4, 10, 11, 12):
          l.append(r[cidx])
        else:
          vec = []
          if r[cidx] != '?':
            vec = vectDict[r[cidx]]
          else:
            vec = get0Embedding(cidx)
          for v in vec:
            l.append(v)
      labeled_data.append(l)
    labeled_data = np.array(labeled_data)

    labels = np.array([1 if r[14] == 1 else -1 for r in raw_training])
    
    new_data = []
    for r in raw_testing:
      l = [1]
      for cidx in range(14):
        if cidx in (0, 2, 4, 10, 11, 12):
          l.append(r[cidx])
        else:
          vec = []
          if r[cidx] != '?':
            vec = vectDict[r[cidx]]
          else:
            vec = get0Embedding(cidx)
          for v in vec:
            l.append(v)
      new_data.append(l)
    new_data = np.array(new_data)
    
    def rfunc(t, args):
        (r_0, a) = args
        return r_0/(1+(r_0*t)/a)
    
    # Find Hyperparameters
    if False:
        N = len(labeled_data)
        k = 5
        idxs = np.array(range(len(labeled_data)))
        bestvalacc = 0
        (bestepochs, bestC, bestr_0, besta) = (0, 0, 0, 0)
        # Test epochs
        for t in (5, 10, 25, 50, 100, 500, 1000):
            # test C
            for c in (100/N, 10/N, 1/N, 0.1/N, 0.01/N):
                # test r_0
                for r in (1, 0.1, 0.01, 0.001, 0.0001):
                    # test a
                    for atest in (100, 10, 1, 0.1, 0.01, 0.001):
                        # k-fold cross validation
                        np.random.shuffle(idxs)
                        idxslices = np.array_split(idxs, k)
                        summedvalaccs = 0
                        for ki in range(k):
                            tridxs = np.concatenate((*idxslices[0:ki], *idxslices[ki+1:]))
                            train_data = labeled_data[tridxs]
                            train_labels = labels[tridxs]
                            
                            validxs = idxslices[ki] 
                            val_data = labeled_data[validxs]
                            val_labels = labels[validxs]
                            
                            w = np.zeros(train_data.shape[1])
                            model = SVM(w)
                            SVMSGD(train_data, train_labels, model, t, c, rfunc, (r, atest))
                            
                            summedvalaccs += AverageError(val_data, val_labels, model)
                        avgvalacc = summedvalaccs / k
                        if avgvalacc > bestvalacc:
                            (bestvalacc, bestepochs, bestC, bestr_0, besta) = (avgvalacc, t, c, r, atest)
                            print('New best validation:', avgvalacc)
                            print('\tparams:', t, c, r, atest)
        (epochs, C, r_0, a) = (bestepochs, bestC, bestr_0, besta)
        print('Best determined validation:', bestvalacc)
        print('Chosen params:')
        print('\tepochs:', bestepochs)
        print('\tC:', bestC)
        print('\tr_0:', bestr_0)
        print('\ta:', besta)
                                
    else:
        # If not looking for parameters, then these will be used
        # These current ones aren't the greatest, but due to time constraints I'm not able to find better ones
        epochs = 5
        C = 0.0004
        r_0 = 0.01
        a = 0.01
    
    
    # Train the model
    w = np.zeros(labeled_data.shape[1])
    model = SVM(w)
    SVMSGD(labeled_data, labels, model, epochs, C, rfunc, (r_0, a))
    print('Soft SVM Average Error: ' + str(AverageError(labeled_data, labels, model)))
    # Generate the output
    numsamples = new_data.shape[0]
    idxs = np.array(range(numsamples))
    output = classif.PredictLabel(new_data)
    output = [1 if d == 1 else 0 for d in output]  # convert labels to {0, 1}
    SaveCSV([[int(i), output[int(i-1)]] for i in idxs], 'Outputs/Kaggle/Soft_SVM_output.csv', ['ID', 'Prediction'])
    
    























             
                        