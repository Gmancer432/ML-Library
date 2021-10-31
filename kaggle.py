
from Util.util import *
from KaggleAttempts import RandomForestAtmpt, LinearRegressionAtmpt, LinearClassificationAtmpt, RandomForestAtmpt2

# Note: the first row of this data is the header
# Only the training data has labels
rawtraining = ReadCSV('Data/kaggle data/train_final.csv')
rawtesting = ReadCSV('Data/kaggle data/test_final.csv')
# Remove unnecessary rows/columns
rawtraining = rawtraining[1:]
rawtesting = [d[1:] for d in rawtesting[1:]]


# Random Forest
if False:
    from KaggleAttempts import RandomForestAtmpt
    RandomForestAtmpt.main(rawtraining, rawtesting)


# Linear Regression
if False:
    from KaggleAttempts import LinearRegressionAtmpt
    LinearRegressionAtmpt.main(rawtraining, rawtesting)

# Linear Classification
if False:
    from KaggleAttempts import LinearClassificationAtmpt
    LinearClassificationAtmpt.main(rawtraining, rawtesting)

# Random Forest with extra pre-processing
# Categorical attributes are converted to vectors of boolean attributes
if True:
    from KaggleAttempts import RandomForestAtmpt2
    RandomForestAtmpt2.main(rawtraining, rawtesting)
