
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
    
    # Prepare attributes
    atrsDict = {'age'            : Attribute('age',            0, [], True),
                'workclass'      : Attribute('workclass',      1, ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?']),
                'fnlwgt'         : Attribute('fnlwgt',         2, [], True),
                'education'      : Attribute('education',      3, ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?']),
                'education-num'  : Attribute('education-num',  4, [], True),
                'marital-status' : Attribute('marital-status', 5, ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', '?']),
                'occupation'     : Attribute('occupation',     6, ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?']),
                'relationship'   : Attribute('relationship',   7, ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?']),
                'race'           : Attribute('race',           8, ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?']),
                'sex'            : Attribute('sex',            9, ['Female', 'Male', '?']),
                'capital-gain'   : Attribute('capital-gain',   10, [], True),
                'capital-loss'   : Attribute('capital-loss',   11, [], True),
                'hours-per-week' : Attribute('hours-per-week', 12, [], True),
                'native-country' : Attribute('native-country', 13, ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']),
                'label'          : Attribute('label',          14, ['1', '0'])
                }
    
    # Note: the currently submitted version of this one had 'education' in this list twice.
    # I have not yet ran it again now that the duplicate is removed.
    atrsList = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    
    lparams = (atrsDict, atrsList, Entropy, None, 4)
    
    # Convert continuous columns
    for t in (trdata, tstdata):
        for c in (0, 2, 4, 10, 11, 12):
            ConvertColumn(t, c, float)
    
    
    # Train the classifier
    T = 500
    m = 1000
    (baggedclassif, errs) = bagging(trdata, 14, ID3, lparams, T, m, giveFinalErrors=True, giveSingleErrors=False, testdata=None)
    
    # Generate the output
    outs = []
    idx = 1
    for d in tstdata:
        outs.append([idx, baggedclassif.PredictLabel(d)])
        idx += 1
    SaveCSV(outs, 'Outputs/Kaggle/RFoutput.csv', ['ID', 'Prediction'])
    
    # Display a graph of the loss
    x = np.linspace(1, T, num=T)
    plt.plot(x, errs)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    