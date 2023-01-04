# -*- coding: utf-8 -*-
pip install river

pip install numba

from river import tree
from river import datasets
from river import ensemble
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/sea_stream.csv")

model = tree.HoeffdingTreeClassifier(
     grace_period=100,
 )

# Creating the committee

# Committee members (Usually online Bagging and Boosting is used)
#1. Adaptive Random Forests
#2. Bagging Classifier - Logistic regression
#3. Leverage Bagging Classifier -Logistic regression
#4. AdaBoost Classifier
#5. Bagging Classifier - Hoeffding trees
#6. Leverage Bagging Classifier -Hoeffding trees

# Adaptive Random Forests model
model_1 = ensemble.AdaptiveRandomForestClassifier(seed=8, leaf_prediction="mc")

from river import preprocessing
from river import linear_model
# Bagging Classifier
model_2 = ensemble.BaggingClassifier(
    model=(preprocessing.StandardScaler()|
       linear_model.LogisticRegression()
   ),n_models=3,
     seed=42)

#Leverage Bagging Classifier
model_3= ensemble.LeveragingBaggingClassifier(
    model=(
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression()
    ),
    n_models=3,
    seed=42)

#AdaBoost Classifier
model_4 =  ensemble.AdaBoostClassifier(
    model=(
    tree.HoeffdingTreeClassifier(
    split_criterion='gini',
    grace_period=2000
    )
    ),
    n_models=5,
    seed=42
    )

# Bagging Classifier
model_5 = ensemble.BaggingClassifier(
    model=(
    tree.HoeffdingTreeClassifier(
    split_criterion='gini',
    grace_period=2000
    )
    ),n_models=3,
     seed=42)

#Leverage Bagging Classifier
model_6= ensemble.LeveragingBaggingClassifier(
    model=(
    tree.HoeffdingTreeClassifier(
    split_criterion='gini',
    grace_period=2000
    )
    ),
    n_models=3,
    seed=42)

def voteEntropy(committee):
  import math
  # converting our list to filter list
  committeeUnique = [ x for i, x in enumerate(committee) if x not in committee[:i]]
  statistic = []
  for i in committeeUnique:
    val = (committee.count(i)/len(committee))*math.log(committee.count(i)/len(committee))
    statistic.append(val)
  return(-sum(statistic))

Committee = [model_1,model_2,model_3,model_4,model_5,model_6]

def DensityWeightedQueryByCommitteePlot(df,threshold,model,Committee,Beta):
  import math
  from statistics import mean
  column_names = []
  for col in df.columns:
    column_names.append(col)
  z = []
  for i in range(df.shape[0]):
    z.append((list(df.loc[i][0:df.shape[1]-1]),df.loc[i][df.shape[1]-1]))
  
  training_data_vec = [] 
  prob1 = []
  prob0 = []
  diff = []
  acc=[]
  correct_cnt = 0
  training_data = 2000
  total_data = 0
  model_1 = Committee[0] 
  model_2 = Committee[1]
  model_3 = Committee[2]
  model_4 = Committee[3]
  model_5 = Committee[4]
  model_6 = Committee[5]
  for x in range(len(z)):
      committee=[]
      a = {}
      data = z[x][0]
      for p in range(len(column_names)-1):
        a[column_names[p]]=data[p]
      b = z[x][1]
      total_data+= 1 
      if x<2000:
          model=model.learn_one(a,b)
          # Training the Committee using initial data
          model_1 = model_1.learn_one(a,b) # Adaptive Random Forest Classifer
          model_2 = model_2.learn_one(a,b) # Bagging Classifier
          model_3 = model_3.learn_one(a,b) # Leverage Bagging Classifier
          model_4 = model_4.learn_one(a,b) # Ada Boost Classifier
          model_5 = model_5.learn_one(a,b) # Bagging Classifier HT
          model_6 = model_6.learn_one(a,b) # Leverage Bagging Classifier HT
          training_data_vec.append(a)
      else:
          pred = model.predict_one(a)
          if pred == b:
              acc.append(1)
              correct_cnt += 1
          else:
              acc.append(0)
          # Making predictions Using Committees
          model_1_pred = model_1.predict_one(a)# Adaptive Random Forest Classifer
          committee.append(model_1_pred)
          model_2_pred  = model_2.predict_one(a) # Bagging Classifier
          if model_2_pred =="True":
              committee.append(1.0) 
          else: 
              committee.append(0.0)
          model_3_pred  = model_3.predict_one(a)#Leverage Bagging Classifier
          if model_3_pred =="True":
              committee.append(1.0)
          else: 
              committee.append(0.0)
          model_4_pred = model_4.predict_one(a) # Ada boost Classifier
          committee.append(model_4_pred)
          model_5_pred_HT  = model_5.predict_one(a) # Bagging Classifier HT
          if model_5_pred_HT=="True":
              committee.append(1.0) 
          else: 
              committee.append(0.0)
          model_6_pred_HT  = model_6.predict_one(a) #Leverage Bagging Classifier HT
          if model_6_pred_HT =="True":
              committee.append(1.0)
          else: 
              committee.append(0.0)
          # Distance measure calculation
          distance_measure = []
          val1 = list(a.values())
          for m in training_data_vec[-500:]:
            val2 = list(m.values())
            distance = math.dist(val1,val2)
            distance_measure.append(distance)
          dist = mean(distance_measure)


          if  voteEntropy(committee)* dist**(Beta)>= threshold:
              model=model.learn_one(a,b)
              training_data += 1
              training_data_vec.append(a)

  from statistics import mean
  import math
  accuracy_measure=[]
  for p in range(math.floor(len(z)/1000)):
    accuracy_measure.append(mean(acc[1:(p+1)*1000]))
  import matplotlib.pyplot as plt

  k = range(1,math.floor(len(z)/1000)+1)
  plt.figure(figsize=(10,10))
  plt.plot(k,accuracy_measure,label="Uncertainity Sampling")
  return(plt)

DensityWeightedQueryByCommitteePlot(df,2.5,model,Committee,0.5)

def DensityWeightedQueryByCommittee(df,threshold,model,Committee,Beta):
  import math
  from statistics import mean
  column_names = []
  for col in df.columns:
    column_names.append(col)
  z = []
  for i in range(df.shape[0]):
    z.append((list(df.loc[i][0:df.shape[1]-1]),df.loc[i][df.shape[1]-1]))
  
  training_data_vec = [] 
  prob1 = []
  prob0 = []
  diff = []
  acc=[]
  correct_cnt = 0
  training_data = 2000
  total_data = 0
  model_1 = Committee[0] 
  model_2 = Committee[1]
  model_3 = Committee[2]
  model_4 = Committee[3]
  model_5 = Committee[4]
  model_6 = Committee[5]
  for x in range(len(z)):
      committee=[]
      a = {}
      data = z[x][0]
      for p in range(len(column_names)-1):
        a[column_names[p]]=data[p]
      b = z[x][1]
      total_data+= 1 
      if x<2000:
          model=model.learn_one(a,b)
          # Training the Committee using initial data
          model_1 = model_1.learn_one(a,b) # Adaptive Random Forest Classifer
          model_2 = model_2.learn_one(a,b) # Bagging Classifier
          model_3 = model_3.learn_one(a,b) #Leverage Bagging Classifier
          model_4 = model_4.learn_one(a,b) # Ada Boost Classifier
          model_5 = model_5.learn_one(a,b) # Bagging Classifier HT
          model_6 = model_6.learn_one(a,b) #Leverage Bagging Classifier HT
          training_data_vec.append(a)
      else:
          pred = model.predict_one(a)
          if pred == b:
              acc.append(1)
              correct_cnt += 1
          else:
              acc.append(0)
          # Making predictions Using Committees
          model_1_pred = model_1.predict_one(a)# Adaptive Random Forest Classifer
          committee.append(model_1_pred)
          model_2_pred  = model_2.predict_one(a) # Bagging Classifier
          if model_2_pred =="True":
              committee.append(1.0) 
          else: 
              committee.append(0.0)
          model_3_pred  = model_3.predict_one(a)#Leverage Bagging Classifier
          if model_3_pred =="True":
              committee.append(1.0)
          else: 
              committee.append(0.0)
          model_4_pred = model_4.predict_one(a) # Ada boost Classifier
          committee.append(model_4_pred)
          model_5_pred_HT  = model_5.predict_one(a) # Bagging Classifier HT
          if model_5_pred_HT=="True":
              committee.append(1.0) 
          else: 
              committee.append(0.0)
          model_6_pred_HT  = model_6.predict_one(a) #Leverage Bagging Classifier HT
          if model_6_pred_HT =="True":
              committee.append(1.0)
          else: 
              committee.append(0.0)
          # Distance measure calculation
          distance_measure = []
          val1 = list(a.values())
          for m in training_data_vec[-500:]:
            val2 = list(m.values())
            distance = math.dist(val1,val2)
            distance_measure.append(distance)
          dist = mean(distance_measure)



          if  voteEntropy(committee)* dist**(Beta)>= threshold:
              model=model.learn_one(a,b)
              training_data += 1
              training_data_vec.append(a)

  from statistics import mean
  import math
  accuracy_measure=[]
  for p in range(math.floor(len(z)/1000)):
    accuracy_measure.append(mean(acc[1:(p+1)*1000]))
  
  return(mean(accuracy_measure))

model = tree.HoeffdingTreeClassifier(
     grace_period=100,
 )

DensityWeightedQueryByCommittee(df,2.5,model,Committee,0.5)









