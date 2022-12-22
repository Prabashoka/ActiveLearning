# -*- coding: utf-8 -*-

    https://colab.research.google.com/drive/14x6G_F3PRcXONuSY9KDV0Chw1GnGACnJ
"""

# Machine Learning Models (Scikit multiflow)

pip install scikit-multiflow

# Imports
from skmultiflow.data.file_stream import FileStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.evaluation import EvaluatePrequential
import pandas as pd
import seaborn as sns
import numpy as np

model = HoeffdingTreeClassifier()

stream = FileStream("https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/sea_stream.csv")

def streamingMlModel(stream,model):
  from skmultiflow.evaluation import EvaluatePrequential
  evaluator = EvaluatePrequential(pretrain_size=200,
                                max_samples=40000, batch_size=2,
                                output_file='results.txt'
                               )
  evaluator.evaluate(stream=stream, model= model)

def file_text_removal(filename):
  # importing regex module
  import re

  # defining object file1 to open
  # results file in read mode
  file1 = open(filename,'r')

  # defining object file2 to open
  # resultsUpdated file in
  # write mode
  file2 = open('resultsUpdated.txt','w')

  # reading each line from original
  # text file
  for line in file1.readlines():

	# reading all lines that begin
	# with "#"
	  x = re.findall("^#", line)
	
	  if not x:
		# printing those lines
		  
		
		# storing only those lines that
		# do not begin with "TextGenerator"
		  file2.write(line)
		
  # close and save the files
  file1.close()
  file2.close()

def MachineLearningAccuracy(stream,model):
  streamingMlModel(stream,model)
  file_text_removal(filename = 'results.txt')
  import pandas as pd
  data=pd.read_table('resultsUpdated.txt', delimiter = ',')
  Id = data['id']
  accuracy = data['mean_acc_[M0]']
  import matplotlib.pyplot as plt
  plt.figure(figsize=(10,10))
  plt.plot(Id,accuracy,label="Machine Learning model")
  return(plt)
