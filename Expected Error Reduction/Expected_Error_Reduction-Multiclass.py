# -*- coding: utf-8 -*-

#  There's no change for the function in the binary classification problems as well as multiclass classification problems.
#  Therefore the same function is used.
pip install river

from river import evaluate
from river import metrics
from river import tree
from river import datasets
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/Prabashoka/ActiveLearning/main/Research%20Data/TimeStampIris.csv')

model = tree.HoeffdingTreeClassifier(
        grace_period=100,
        )

def ExpectedErrorRedWithPlot(df, thresh,model):
  column_names = []
  for col in df.columns:
    column_names.append(col)
  z = []
  for i in range(df.shape[0]):
    z.append((list(df.loc[i][0:df.shape[1]-1]),df.loc[i][df.shape[1]-1]))
  prob1 = []
  prob0 = []
  diff = []
  acc=[]
  correct_cnt = 0
  training_data = 20
  total_data = 0
  for x in range(len(z)):
      a = {}
      data = z[x][0]
      for p in range(len(column_names)-1):
        a[column_names[p]]=data[p]
      b = z[x][1]
      total_data+= 1 
      if x<20:
          final_model = model.learn_one(a,b)
          model=final_model
      else:
          pred = model.predict_one(a)
          if pred == b:
              acc.append(1)
              correct_cnt += 1
          else:
              acc.append(0)
          prob_0_theta  = model.predict_proba_one(a)[0]
          prob_1_theta  = model.predict_proba_one(a)[1]
          model = model.learn_one(a,pred)
          prob_pred_theta_plus = model.predict_proba_one(a)
          value  = 1-max(prob_pred_theta_plus.values())
          if(value>thresh):
            final_model=final_model.learn_one(a,b)
            training_data += 1
            model = final_model
  from statistics import mean
  import math
  accuracy_measure=[]
  for p in range(math.floor(len(z)/10)):
    accuracy_measure.append(mean(acc[1:(p+1)*10]))
  import matplotlib.pyplot as plt

  k = range(1,math.floor(len(z)/10)+1)
  plt.figure(figsize=(10,10))
  plt.plot(k,accuracy_measure,label="Uncertainity Sampling")
  return(plt)

ExpectedErrorRedWithPlot(df, 0.4,model)


def ExpectedErrorRed(df, thresh,model):
  column_names = []
  for col in df.columns:
    column_names.append(col)
  z = []
  for i in range(df.shape[0]):
    z.append((list(df.loc[i][0:df.shape[1]-1]),df.loc[i][df.shape[1]-1]))
  prob1 = []
  prob0 = []
  diff = []
  acc=[]
  correct_cnt = 0
  training_data = 20
  total_data = 0
  for x in range(len(z)):
      a = {}
      data = z[x][0]
      for p in range(len(column_names)-1):
        a[column_names[p]]=data[p]
      b = z[x][1]
      total_data+= 1 
      if x<20:
          final_model = model.learn_one(a,b)
          model=final_model
      else:
          pred = model.predict_one(a)
          if pred == b:
              acc.append(1)
              correct_cnt += 1
          else:
              acc.append(0)
          prob_0_theta  = model.predict_proba_one(a)[0]
          prob_1_theta  = model.predict_proba_one(a)[1]
          model = model.learn_one(a,pred)
          prob_pred_theta_plus = model.predict_proba_one(a)
          value  = 1-max(prob_pred_theta_plus.values())
          if(value>thresh):
            final_model=final_model.learn_one(a,b)
            training_data += 1
            model = final_model
  from statistics import mean
  import math
  accuracy_measure=[]
  for p in range(math.floor(len(z)/10)):
    accuracy_measure.append(mean(acc[1:(p+1)*10]))
  return(accuracy_measure)



