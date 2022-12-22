# -*- coding: utf-8 -*-
pip install river

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import re

data = pd.read_csv("https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/elec.csv")

data_stream = data["class"]

import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_data(data_stream, drifts=None):
    fig = plt.figure(figsize=(14,3), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])
    ax1.grid()
    ax1.plot(data_stream, label='Stream')
    ax2.grid(axis='y')
    ax2.hist(data_stream, label=r'$data_stream$')
    if drifts is not None:
        for drift_detected in drifts:
            ax1.axvline(drift_detected, color='red')
    plt.show()

def ADWINDetect(data_stream):
  from river import drift
  drift_detector = drift.ADWIN()
  drifts = []
  for i, val in enumerate(data_stream):
    drift_detector.update(val)   # Data is processed one sample at a time
    if drift_detector.drift_detected:
        # The drift detector indicates after each sample if there is a drift in the data
        drifts.append(i)
  plot_data(data_stream, drifts)

ADWINDetect(data_stream)

def DDMDetect(data_stream):
  from river import drift
  drift_detector = drift.DDM()
  drifts = []
  for i, val in enumerate(data_stream):
    drift_detector.update(val)   # Data is processed one sample at a time
    if drift_detector.drift_detected:
        # The drift detector indicates after each sample if there is a drift in the data
        drifts.append(i)
  plot_data(data_stream, drifts)

DDMDetect(data_stream)

def PageHinkleyDetect(data_stream):
  from river import drift
  drift_detector = drift.PageHinkley()
  drifts = []
  for i, val in enumerate(data_stream):
    drift_detector.update(val)   # Data is processed one sample at a time
    if drift_detector.drift_detected:
        # The drift detector indicates after each sample if there is a drift in the data
        drifts.append(i)
  plot_data(data_stream, drifts)

PageHinkleyDetect(data_stream)

def EarlyDriftDetect(data_stream):
  from river import drift
  drift_detector = drift.EDDM()
  drifts = []
  for i, val in enumerate(data_stream):
    drift_detector.update(val)   # Data is processed one sample at a time
    if drift_detector.drift_detected:
        # The drift detector indicates after each sample if there is a drift in the data
        drifts.append(i)
  plot_data(data_stream, drifts)

EarlyDriftDetect(data_stream)

def KSWINDetect(data_stream):
  from river import drift
  drift_detector = drift.KSWIN()
  drifts = []
  for i, val in enumerate(data_stream):
    drift_detector.update(val)   # Data is processed one sample at a time
    if drift_detector.drift_detected:
        # The drift detector indicates after each sample if there is a drift in the data
        drifts.append(i)
  plot_data(data_stream, drifts)

KSWINDetect(data_stream)

def HDDM_WDetect(data_stream):
  from river import drift
  drift_detector = drift.HDDM_W()
  drifts = []
  for i, val in enumerate(data_stream):
    drift_detector.update(val)   # Data is processed one sample at a time
    if drift_detector.drift_detected:
        # The drift detector indicates after each sample if there is a drift in the data
        drifts.append(i)
  plot_data(data_stream, drifts)

HDDM_WDetect(data_stream)

def HDDM_ADetect(data_stream):
  from river import drift
  drift_detector = drift.HDDM_A()
  drifts = []
  for i, val in enumerate(data_stream):
    drift_detector.update(val)   # Data is processed one sample at a time
    if drift_detector.drift_detected:
        # The drift detector indicates after each sample if there is a drift in the data
        drifts.append(i)
  plot_data(data_stream, drifts)

HDDM_ADetect(data_stream)

