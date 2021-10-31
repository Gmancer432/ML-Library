
from Util.util import SaveCSV, ConvertColumn
from DecisionTree.ID3 import Attribute, Entropy, ID3
from EnsembleLearning.bagging import bagging, BaggedClassif
import numpy as np
import matplotlib.pyplot as plt


#### Runs the data through a random forest

def main(raw_training, raw_testing):
    # Prepare the data
    trdata = raw_training
    tstdata = raw_testing
    
    # Convert continuous columns
    for t in (trdata, tstdata):
        for c in (0, 2, 4, 10, 11, 12):
            ConvertColumn(t, c, float)    
    
    # Prepare attributes
    atrsDictCat = {
                'workclass'      : ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
                'education'      : ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
                'marital-status' : ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
                'occupation'     : ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
                'relationship'   : ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
                'race'           : ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
                'sex'            : ['Female', 'Male'],
                'native-country' : ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],
                }
    
    # Vectorize the categorical attributes
    def makeVector(item, valuelist):
        v = np.zeros(len(valuelist))
        if item in valuelist:
            v[valuelist.index(item)] = 1
        return v.tolist()
        

    trdata = [[d[0],
               *makeVector(d[1], atrsDictCat['workclass']),
               d[2],
               *makeVector(d[3], atrsDictCat['education']),
               d[4],
               *makeVector(d[5], atrsDictCat['marital-status']),
               *makeVector(d[6], atrsDictCat['occupation']),
               *makeVector(d[7], atrsDictCat['relationship']),
               *makeVector(d[8], atrsDictCat['race']),
               *makeVector(d[9], atrsDictCat['sex']),
               d[10],
               d[11],
               d[12],
               *makeVector(d[13], atrsDictCat['native-country']),
               d[14]
               ] for d in raw_training ]
    
    labelidx = len(trdata[0]) - 1
    
    tstdata = [[d[0],
                *makeVector(d[1], atrsDictCat['workclass']),
                d[2],
                *makeVector(d[3], atrsDictCat['education']),
                d[4],
                *makeVector(d[5], atrsDictCat['marital-status']),
                *makeVector(d[6], atrsDictCat['occupation']),
                *makeVector(d[7], atrsDictCat['relationship']),
                *makeVector(d[8], atrsDictCat['race']),
                *makeVector(d[9], atrsDictCat['sex']),
                d[10],
                d[11],
                d[12],
                *makeVector(d[13], atrsDictCat['native-country']),
                ] for d in raw_testing ]
    
    
    atrsList = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    
    newatrsList = []
    
    # Construct the atrsDict
    atrsDict = {}
    curidx = 0
    for a in atrsList:
        if a in atrsDictCat:
            curlist = atrsDictCat[a]
            for item in curlist:
                curitem = 'is_' + item
                atrsDict[curitem] = Attribute(curitem, curidx, [], True)
                curidx += 1
                newatrsList.append(curitem)
        else:
            atrsDict[a] = Attribute(a, curidx, [], True)
            curidx += 1
            newatrsList.append(a)
    atrsDict['label'] = Attribute('label', curidx, ['1', '0'])
        
    lparams = (atrsDict, newatrsList, Entropy, None, 4)
    
    
    
    # Train the classifier
    T = 500
    m = 1000
    (baggedclassif, errs) = bagging(trdata, labelidx, ID3, lparams, T, m, giveFinalErrors=True, giveSingleErrors=False, testdata=None)
    
    # Generate the output
    outs = []
    idx = 1
    for d in tstdata:
        outs.append([idx, baggedclassif.PredictLabel(d)])
        idx += 1
    SaveCSV(outs, 'Outputs/Kaggle/RF_extraproc_output.csv', ['ID', 'Prediction'])
    
    # Display a graph of the loss
    x = np.linspace(1, T, num=T)
    plt.plot(x, errs)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    