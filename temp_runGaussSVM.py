
from Util.util import *

# Note: the first row of this data is the header
# Only the training data has labels
rawtraining = ReadCSV('Data/kaggle data/train_final.csv')
rawtesting = ReadCSV('Data/kaggle data/test_final.csv')
# Remove unnecessary rows/columns
rawtraining = rawtraining[1:]
rawtesting = [d[1:] for d in rawtesting[1:]]


# SVM Dual with Gaussian Kernel
# This one takes too much memory and time to do fully
if True:
    from KaggleAttempts import SVMGaussAtmpt
    SVMGaussAtmpt.main(rawtraining, rawtesting)
