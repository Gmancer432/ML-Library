
'''
    This attempt was developed in Google Colab, and copied over to here to be consistent with the other attempts.
    Google Colab link: https://colab.research.google.com/drive/1EC1h--DKdDVCeFCzkL8npButiHnWE89p?usp=sharing
    If you're concerned about revision history, you can open the link and to go File -> 'Revision History' to see the history of my edits.
'''

import numpy as np
import matplotlib.pyplot as plt
import math
from Util.util import *
# NN imports
import torch
import torch.nn as nn
import torch.optim as optim

### Runs the data through Linear Regression

def main():
    """# Importing and Processing Data
    training data [N, dim]: labeled_data

    training labels [N]: labels

    testing data [N, dim]: new_data
    """

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

    # import data
    # remember to upload the files
    # The first row is the column labels

    rawtrainingdata = ReadCSV('Data/kaggle data/train_final.csv')[1:]
    # columns 0, 2, 4, 10, 11, 12, and 14 in the training data are numerical columns
    for i in (0, 2, 4, 10, 11, 12, 14):
        ConvertColumn(rawtrainingdata, i, float)

    # convert to vectorized nparray
    labeled_data = []
    for r in rawtrainingdata:
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

    labels = np.array([r[14] for r in rawtrainingdata])

    rawtestingdata = ReadCSV('Data/kaggle data/test_final.csv')[1:]
    # columns 0, 2, 4, 10, 11, 12, and 14 in the training data are numerical columns
    for i in (1, 3, 5, 11, 12, 13):
        ConvertColumn(rawtestingdata, i, float)

    # convert to vectorized nparray
    new_data = []
    for r in rawtestingdata:
      l = []
      for cidx in range(1, 15):
        if cidx in (1, 3, 5, 11, 12, 13):
          l.append(r[cidx])
        else:
          vec = []
          if r[cidx] != '?':
            vec = vectDict[r[cidx]]
          else:
            vec = get0Embedding(cidx-1)
          for v in vec:
            l.append(v)
      new_data.append(l)
    new_data = np.array(new_data)

    """# Neural Networks"""

    dtype=torch.double
    device = torch.device('cpu')

    """## Functions"""

    def AvgAccuracy(model, data, labels): 
      model = model.to(device=device)
      data = data.to(device=device)
      labels = labels.to(device=device)
      num_correct = 0
      num_samples = 0
      model.eval()  # set model to evaluation mode
      with torch.no_grad():
        preds = model(data).round()
        num_samples = preds.size(0)
        num_correct = (preds == labels).sum()
        acc = float(num_correct) / num_samples
        return acc

    def trainNNStochastic(model, optimizer, train_data, train_labels, epochs, val_data, val_labels):
      idxs = np.array(range(len(train_data)))
      t = 0
      print_every = 10000
      for e in range(epochs):
        np.random.shuffle(idxs)
        for i in idxs:  
          model.train() # put model to training mode
          x = train_data[i]
          y = train_labels[i]
          score = model(x)
          loss = torch.nn.functional.mse_loss(score, y) # mean squared-error loss

          # Zero out all the previous gradients
          optimizer.zero_grad()

          # Compute the current gradients
          loss.backward()

          # Update the weights according to the optimizer
          optimizer.step()
          if t % print_every == 0:
            print('Step ' + str(t))
            print('\tLoss: ' + str(loss.item()))
            if val_data != None:
              print('\tAccuracy: ' + str(AvgAccuracy(model, val_data, val_labels)))
          t += 1

    def trainNNFull(model, optimizer, train_data, train_labels, epochs, val_data, val_labels):
      model = model.to(device=device)
      train_data = train_data.to(device=device)
      train_labels = train_labels.to(device=device)
      if val_data != None:
        val_data = val_data.to(device=device)
        val_labels = val_labels.to(device=device)
      t = 0
      print_every = 100
      for e in range(epochs):
        model.train() # put model to training mode
        score = model(train_data)
        loss = torch.nn.functional.mse_loss(score, train_labels) # mean squared-error loss

        # Zero out all the previous gradients
        optimizer.zero_grad()

        # Compute the current gradients
        loss.backward()

        # Update the weights according to the optimizer
        optimizer.step()
        if t % print_every == 0:
          print('Step ' + str(t))
          print('\tLoss: ' + str(loss.item()))
          if val_data != None:
            print('\tAccuracy: ' + str(AvgAccuracy(model, val_data, val_labels)))
        t += 1

    def KFoldCrossValidation(model, optimizer, train_data, train_labels, epochs):
      model.reset_parameters()
      raise NotImplementedError

    # Creates a multi-layer fully-connected neural network with sigmoid activations
    def ModelMaker(datalen, layerlengths, outputlen):
      layers = []
      if len(layerlengths) == 0:
        return [nn.Linear(datalen, outputlen)]
      
      layers.append(nn.Linear(datalen, layerlengths[0], dtype=dtype))
      layers.append(nn.Sigmoid())
      for i in range(1, len(layerlengths)):
        layers.append(nn.Linear(layerlengths[i-1], layerlengths[i], dtype=dtype))
        layers.append(nn.Sigmoid())
      layers.append(nn.Linear(layerlengths[-1], outputlen, dtype=dtype))
      layers.append(nn.Sigmoid()) # Bind the output to [0, 1]
      return layers

    """## Data"""

    nn_labeled_data = torch.tensor(labeled_data, dtype=dtype, device=device)
    nn_labels = torch.tensor(labels, dtype=dtype, device=device).reshape((len(labeled_data), 1))
    nn_new_data = torch.tensor(new_data, dtype=dtype, device=device)

    """## Models"""

    datalen = len(labeled_data[0])
    params = (datalen*10, datalen*10)
    model = nn.Sequential(*ModelMaker(datalen, params, 1))
    lr = 0.1
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_val = round(len(nn_labeled_data) / 5)
    epochs = 100000

    """## Training and Results"""

    trainNNStochastic(model, optimizer, nn_labeled_data[:-num_val], nn_labels[:-num_val], epochs, nn_labeled_data[-num_val:], nn_labels[-num_val:])

    print('NN Accuracy:', AvgAccuracy(model, nn_labeled_data, nn_labels))

    """## Predictions on New Data"""

    # Train the model on the full labeled data using the selected parameters
    trainNNStochastic(model, optimizer, nn_labeled_data, nn_labels, epochs, None, None)

    # Generate the output
    nn_new_data = nn_new_data.to(device=device)
    numsamples = new_data.shape[0]
    idxs = np.array(range(1, numsamples+1))
    output = model(nn_new_data)
    output = [1 if d == 1 else 0 for d in output]  # convert labels to {0, 1}
    SaveCSV([[int(i), output[int(i-1)]] for i in idxs], 'Outputs/Kaggle/NN_output.csv', ['ID', 'Prediction'])
    
    























             
                        