
from Util.util import ConvertColumn, SaveCSV
from LinearRegression.GradientDescent import GradientDescent
from LinearRegression.Linear import LMS
import numpy as np
import matplotlib.pyplot as plt

### Runs the data through Linear Regression

def main(raw_training, raw_testing):
    # Prepare the data
    # Categorical attributes are converted into vectors of boolean attributes
    
    # Convert the columns that are already numbers
    for t in (raw_training, raw_testing):
        for c in (0, 2, 4, 10, 11, 12):
            ConvertColumn(t, c, float)
    
    atrsDict = {
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
        return v
        
    def expandData(indata):
        outdata = np.array([[1,
                        d[0],
                        *makeVector(d[1], atrsDict['workclass']),
                        d[2],
                        *makeVector(d[3], atrsDict['education']),
                        d[4],
                        *makeVector(d[5], atrsDict['marital-status']),
                        *makeVector(d[6], atrsDict['occupation']),
                        *makeVector(d[7], atrsDict['relationship']),
                        *makeVector(d[8], atrsDict['race']),
                        *makeVector(d[9], atrsDict['sex']),
                        d[10],
                        d[11],
                        d[12],
                        *makeVector(d[13], atrsDict['native-country'])
                        ] for d in indata ])
        return outdata
    
    trdata = expandData(raw_training)
    tstdata = expandData(raw_testing)
    
    # Convert labels to {-1, 1}
    trlabels = np.array([1 if d[14] == '1' else -1 for d in raw_training])
    
    # Train the Linear model
    classif = LMS(np.zeros(trdata.shape[1]), classifier=True)
    batchsize = 1000
    r = 0.0000000000000000001
    T = 1000000
    (classif, errs) = GradientDescent(trdata, trlabels, classif, batchsize, r, tolerance=None, T=T, giveTrainingLoss=True, giveTestingLoss=False, testingdata=None, testinglabels=None)
    
    # Generate the output
    numsamples = tstdata.shape[0]
    idxs = np.linspace(1, numsamples, num=numsamples)
    output = classif.PredictLabel(tstdata)
    output = [1 if d == 1 else 0 for d in output]  # convert labels to {0, 1}
    SaveCSV([[int(i), output[int(i-1)]] for i in idxs], 'Outputs/Kaggle/LMSClassif_GD_output.csv', ['ID', 'Prediction'])
    
    # Show a plot of the training loss
    plt.plot(np.linspace(0, T, num=T+1), errs)
    plt.show()
    
    























             
                        