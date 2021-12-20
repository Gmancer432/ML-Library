
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
      l = []
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
      l = []
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
    
    # Train the model
    variance = labeled_data.var(axis=0).mean()
    model = SVMDual(GaussianKernel, variance)
    C = 1 / len(labeled_data)
    model.Optimize(labeled_data, labels, C)
    print('SVM Gauss Error: ' + str(AverageError(labeled_data, labels, model)))
    # Generate the output
    numsamples = new_data.shape[0]
    idxs = np.array(range(numsamples))
    output = classif.PredictLabel(new_data)
    output = [1 if d == 1 else 0 for d in output]  # convert labels to {0, 1}
    SaveCSV([[int(i), output[int(i-1)]] for i in idxs], 'Outputs/Kaggle/SVM_Gauss_output.csv', ['ID', 'Prediction'])
    
    























             
                        