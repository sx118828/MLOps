!pip install lightgbm

import mlflow
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

