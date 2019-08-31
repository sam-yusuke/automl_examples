#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
# Read dataset
df = pd.read_csv('data/balance_scale/balance-scale.data', 
                 names=['balance', 'var1', 'var2', 'var3', 'var4'])

# Display example observations
df.head()

ax = sns.countplot(x="balance", data=df)


#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Separate input features (X) and target variable (y)
y = df.balance
X = df.drop('balance', axis=1)
 
# Train model
clf_0 = LogisticRegression().fit(X, y)
 
# Predict on training set
pred_y_0 = clf_0.predict(X)

print( accuracy_score(pred_y_0, y) )

#%%
f = open('data/balance_scale/balance-scale.data')
f.readline()
data = np.loadtxt(f)
data

#%%
X = X.values
class_names = ['B', 'R', 'L']
y = [class_names.index(c) for c in y]

#%%
import numpy as np
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight    


# X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)
y = np.array(y).astype(np.int64)

class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()

        # self.dense0 = nn.Linear(20, 10)
        self.nonlin = nonlin
        self.output = nn.Linear(4, 3)

    def forward(self, X, **kwargs):
        # X = self.nonlin(self.output(X))
        X = F.softmax(self.output(X), dim=1)
        return X
        

from functools import partial
import skorch
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class DataBalancingCallback(skorch.callbacks.Callback):
    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        # For unbalanced dataset we create a weighted sampler
        weights = make_weights_for_balanced_classes(dataset_train, 3)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(weights))

        net.iterator_train = partial(net.iterator_train, sampler=sampler)

        weights = make_weights_for_balanced_classes(dataset_valid, 3)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(weights))
        net.iterator_valid = partial(net.iterator_valid, sampler=sampler)

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        self.original_iterator_train = net.iterator_train
        self.original_iterator_valid = net.iterator_valid

    def on_train_end(self, net, X=None, y=None, **kwargs):
        net.iterator_train = self.original_iterator_train
        net.iterator_valid = self.original_iterator_valid

cp = skorch.callbacks.Checkpoint(dirname='exp2')

net = NeuralNetClassifier(
    MyModule,
    max_epochs=200,
    lr=0.1,
    batch_size=1024,
    train_split=skorch.dataset.CVSplit(4),
    # criterion=FocalLoss,
    criterion=partial()
    callbacks=[
        DataBalancingCallback(),
        cp,
    ]
)

from sklearn.metrics import accuracy_score

net.fit(X, y)
y_proba = net.predict_proba(X)
net.iterator_valid = torch.utils.data.DataLoader

#%%
import sklearn
	
from sklearn.metrics import roc_auc_score, confusion_matrix
pred_y_2 = net.predict(X)

print(confusion_matrix(y, pred_y_2))
print(roc_auc_score(sklearn.preprocessing.label_binarize(y, [0, 1, 2]), net.predict_proba(X)))

print( np.unique( pred_y_2 ) )
#%%
net.get_default_callbacks()

#%%
cp2 = skorch.callbacks.Checkpoint(dirname='exp2')
another_net = NeuralNetClassifier(
    MyModule,
    max_epochs=30,
    lr=0.1,
    batch_size=1024,
    train_split=skorch.dataset.CVSplit(4),
)
another_net.initialize()
another_net.load_params(checkpoint=cp2)

import sklearn
	
from sklearn.metrics import roc_auc_score, confusion_matrix
pred_y_2 = another_net.predict(X)

print(confusion_matrix(y, pred_y_2))
print(roc_auc_score(sklearn.preprocessing.label_binarize(y, [0, 1, 2]), another_net.predict_proba(X)))


#%%
