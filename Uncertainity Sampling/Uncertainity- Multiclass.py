# -*- coding: utf-8 -*-
"""Untitled

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18NvMf9DZl4RXMkdDresUSPZ2H_T5EqSI
"""

pip install river

from river import evaluate
from river import metrics
from river import tree
from river import datasets
from river import ensemble
import pandas as pd
import math





model = tree.HoeffdingTreeClassifier(
        grace_period=100,
        )

def uncertainitySampWithPlot(df, thresh,model,size):
  column_names = []
  for col in df.columns:
    column_names.append(col)
  z = []
  for i in range(df.shape[0]):
    z.append((list(df.loc[i][0:df.shape[1]-1]),df.loc[i][df.shape[1]-1]))
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
          model=model.learn_one(a,b)
      else:
          pred = model.predict_one(a)
          if pred == b:
              acc.append(1)
              correct_cnt += 1
          else:
              acc.append(0)
          prob_vec = sorted(model.predict_proba_one(a).values(),reverse = True)
          diffBetweeen2 = prob_vec[0]-prob_vec[1]
          diff.append(diffBetweeen2)
          if abs(diffBetweeen2)<= thresh:
              model=model.learn_one(a,b)
              training_data += 1
  from statistics import mean
  import math
  accuracy_measure=[]
  for p in range(math.floor(len(z)/size)):
    accuracy_measure.append(mean(acc[1:(p+1)*size]))
  import matplotlib.pyplot as plt

  k = range(1,math.floor(len(z)/size)+1)
  plt.figure(figsize=(10,10))
  plt.plot(k,accuracy_measure,label="Uncertainity Sampling")
  return(plt)

uncertainitySampWithPlot(df, 0.25,model,10)