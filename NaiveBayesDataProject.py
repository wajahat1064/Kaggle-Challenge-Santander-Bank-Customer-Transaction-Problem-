# -*- coding: utf-8 -*-
"""
Created on Fri May 3 19:18:44 2019

@author: Wajahat Waheed
"""
import numpy as np
import time
import sys
import math
import pandas as pd   
import matplotlib.pyplot as plt
link = 'https://drive.google.com/open?id=1oFPCrtpVq51l50Em78Z0jEtGuJd46_3h'
fluff,id=link.split('=')
import pandas as pd
df3 = pd.read_csv('/Users/mr.laptop/Desktop/Naive Bayes/train.csv')
trainn_features = np.array(df3.iloc[:,2:].as_matrix())
train_labels = np.array(df3.iloc[:,1].as_matrix())
from sklearn.preprocessing import KBinsDiscretizer
enc = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
train_features = enc.fit_transform(trainn_features)
#print(train_features)
zero_class,one_class = 0, 0 
dataset_zeroclass_features = np.zeros((179902, 200))
dataset_zeroclass_labels = np.zeros((179902))
dataset_oneclass_features = np.zeros((20098, 200))
dataset_oneclass_labels = np.zeros((20098))
for i in range(0, train_labels.shape[0]):
  if train_labels[i] == 0:
    dataset_zeroclass_features[zero_class, :] = train_features[i,:]
    dataset_zeroclass_labels[zero_class] = train_labels[i]
    zero_class += 1
  elif train_labels[i] == 1:
    dataset_oneclass_features[one_class, :] = train_features[i, :]
    dataset_oneclass_labels[one_class] = train_labels[i]
    one_class += 1
print('zero_class:', zero_class)
print('one_class: ', one_class)
trainingzero_instances = math.floor(zero_class*.11)
trainingone_instances = math.floor(one_class*0.70)
validationzero_instances = math.floor(zero_class*0.033)
validationone_instances = math.floor(one_class*.20)
testzero_instances = math.floor(zero_class*0.017)
testone_instances = math.floor(one_class*0.10)

training_features = np.vstack((dataset_zeroclass_features[0:trainingzero_instances, :], dataset_oneclass_features[0:trainingone_instances, :]))
training_labels = np.hstack((dataset_zeroclass_labels[0:trainingzero_instances], dataset_oneclass_labels[0:trainingone_instances]))

validation_features = np.vstack((dataset_zeroclass_features[trainingzero_instances:validationzero_instances+trainingzero_instances, :], dataset_oneclass_features[trainingone_instances:validationone_instances+trainingone_instances, :]))
validation_labels = np.hstack((dataset_zeroclass_labels[trainingzero_instances:validationzero_instances+trainingzero_instances], dataset_oneclass_labels[trainingone_instances:validationone_instances+trainingone_instances]))

test_features = np.vstack((dataset_zeroclass_features[validationzero_instances:testzero_instances+validationzero_instances, :], dataset_oneclass_features[validationone_instances:testone_instances+validationone_instances, :]))
test_labels = np.hstack((dataset_zeroclass_labels[validationzero_instances:testzero_instances+validationzero_instances], dataset_oneclass_labels[validationone_instances:testone_instances+validationone_instances]))
one_vector, zero_vector = get_feature_counts(training_features,training_labels)
#print (one_vector, zero_vector)
problone,problzero = probabilityoflikelihood(pos_vector,neg_vector)
#print(problone,problzero)
probpone,probpzero= probabilityofprior(training_labels)
#print(probpone,probpzero)
x,y_pred= finalprobability(test_features,test_labels)
print('Accuracy: '+str(x))
TP, TN, FP, FN = 0, 0, 0, 0
#print(y_pred)
#print(len(y_pred))
for i in range(0, test_labels.shape[0]):
    if test_labels[i] == 0 and y_pred[i] <= 0.5:
        TN += 1
    elif test_labels[i] == 1 and y_pred[i] >= 0.5:
        TP += 1
    elif test_labels[i] == 0 and y_pred[i] >= 0.5:
        FP += 1
    elif test_labels[i] == 1 and y_pred[i] <= 0.5:
        FN += 1
confusion_matrixx = np.array([[TP, FP], [FN, TN]])
print(confusion_matrixx)
precision1 = TP/(TP+FP) 
precision0 = TN/(TN+FN) 
print('macro_precision:', (precision1+precision0)/2) 
print('micro_precision:', (TP+TN)/(TP+FP+TN+FN)) 
recall1 = TP/(TP+FN) 
recall0 = TN/(TN+FP) 
print('macro_recall:', (recall1+recall0)/2) 
print('micro_recall:', (TP+TN)/(TP+TN+FN+FP)) 
micro_Negative_Predictive_value = TN/(TN + FN) 
print('NPV:', micro_Negative_Predictive_value) 
F1 = 2*(precision1*recall1)/(precision1+recall1) 
print('F1', F1) 
micro_F2 = 5 *precision1*recall1/(4*precision1 + recall1) 
print('micro_F2:', micro_F2) 
false_positive_rate = FP/(FP + TN) 
print('FPR:', false_positive_rate) 
false_discovery_rate = FP/(FP + TP) 
print('FDR:',false_discovery_rate )
def finalprobability(test_x,test_y):
    score=0
    words= test_x.shape[1]
    tweets=test_x.shape[0]
    pos_vector= np.zeros((words))
    neg_vector=np.zeros((words))
    y_pred=np.zeros((test_y.shape[0]))
    for i in range(0,tweets):
      finalpp=0
      finalpn=0
      ninf= float('-Inf')
      for j in range(0,words):  
          number = test_x[i,j]
          if np.log(problp[j]) == ninf:
            finalpp+= number*-2000000
          else:
            finalpp +=   number*np.log(problp[j])
          if np.log(probln[j]) == ninf:
            finalpn += number*-2000000
          else:
            finalpn += number*np.log(probln[j])
      if math.log(probpp) == ninf:
         finalpp += -2000000
      else:
        finalpp += math.log(probpp)
      if math.log(probpn) ==ninf:
        finalpn += -2000000
      else:
        finalpn += math.log(probpn)
      vector = np.zeros((2))
      vector[0]= finalpn
      vector[1]= finalpp
      finalvalue = np.argmax(vector)
      y_pred[i]= finalvalue
      if test_y[i] == 0 and y_pred[i]==0:
          score+=1
      if test_y[i] == 1 and y_pred[i]==1:
          score+=1
    print(finalvalue)    
    print(len(test_y))
    return (score/len(test_y)), (y_pred)
#probability of prior
def probabilityofprior(train_y):
  count_positive=0
  count_negative=0
  for i in range(0,len(train_y)):
    if train_y[i] == 1:
      count_positive+= 1
    if train_y[i] == 0:
      count_negative+= 1
  return (count_positive/len(train_y)), (count_negative/len(train_y))
#Function to show the probability of likelihood
def probabilityoflikelihood(pos_vector,neg_vector): 
  denforpositive= np.sum(pos_vector)
  denfornegative=np.sum(neg_vector)
  return pos_vector/denforpositive, neg_vector/ denfornegative
#This function gets the counts of the feature 
def get_feature_counts(train_x, train_y):
    rows_instances= train_x.shape[1]
    features=train_x.shape[0]
    pos_vector= np.zeros((rows_instances))
    neg_vector=np.zeros((rows_instances))
    for i in range(0,rows_instances):
      cnt_pst=0
      cnt_neg=0
      for j in range(0,features):  
          number = train_x[j,i]
          if train_y[j] == 1:
            pos_vector[i] = pos_vector[i] + number
          if train_y[j] == 0:
            neg_vector[i] = neg_vector[i]+ number       
    return pos_vector, neg_vector
