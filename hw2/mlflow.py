!pip install lightgbm # ВНИМАНИЕ при повторном запуске, закомментировать строку!
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

california_housing = fetch_california_housing(as_frame=True)
DF = california_housing.frame

plt.rcParams["figure.figsize"] = (22,14)
plt.suptitle('Диаграммы распределения значений столбцов данных california_housing')

plt.subplot(4, 4, 1)
# гистограммa MedInc
plt.xlabel("MedInc")
plt.ylabel("Frequency")
plt.hist(DF['MedInc'], bins = 50);

plt.subplot(4, 4, 2)
# гистограммa HouseAge     
plt.xlabel("HouseAge")
plt.ylabel("Frequency")
plt.hist(DF['HouseAge'], bins = 50);

plt.subplot(4, 4, 3)
# гистограммa AveRooms
plt.xlabel("AveRooms")
plt.ylabel("Frequency")
plt.hist(DF['AveRooms'], bins = 50);

plt.subplot(4, 4, 4)
# гистограммa AveBedrms    
plt.xlabel("AveBedrms")
plt.ylabel("Frequency")
plt.hist(DF['AveBedrms'], bins = 50);

plt.subplot(4, 4, 5)
# гистограммa Population   
plt.xlabel("Population")
plt.ylabel("Frequency")
plt.hist(DF['Population'], bins = 50);

plt.subplot(4, 4, 6)
# гистограммa AveOccup     
plt.xlabel("AveOccup")
plt.ylabel("Frequency")
plt.hist(DF['AveOccup'], bins = 50);

plt.subplot(4, 4, 7)
# гистограммa Latitude
plt.xlabel("Latitude")
plt.ylabel("Frequency")
plt.hist(DF['Latitude'], bins = 50);

plt.subplot(4, 4, 8)
# гистограммa Longitude    
plt.xlabel("Longitude")
plt.ylabel("Frequency")
plt.hist(DF['Longitude'], bins = 50);

plt.show();

scaler_preproces = QuantileTransformer()
scaler_preproces.fit(california_housing.data)
X_scal = scaler_preproces.transform(california_housing.data)

plt.rcParams["figure.figsize"] = (22,14)
plt.suptitle('Диаграммы распределения значений столбцов данных california_housing после preprocessing')

plt.subplot(4, 4, 1)
# гистограммa MedInc
plt.xlabel("MedInc")
plt.ylabel("Frequency")
plt.hist(pd.DataFrame(X_scal)[0], bins = 50);

plt.subplot(4, 4, 2)
# гистограммa HouseAge     
plt.xlabel("HouseAge")
plt.ylabel("Frequency")
plt.hist(pd.DataFrame(X_scal)[1], bins = 50);

plt.subplot(4, 4, 3)
# гистограммa AveRooms
plt.xlabel("AveRooms")
plt.ylabel("Frequency")
plt.hist(pd.DataFrame(X_scal)[2], bins = 50);

plt.subplot(4, 4, 4)
# гистограммa AveBedrms    
plt.xlabel("AveBedrms")
plt.ylabel("Frequency")
plt.hist(pd.DataFrame(X_scal)[3], bins = 50);

plt.subplot(4, 4, 5)
# гистограммa Population   
plt.xlabel("Population")
plt.ylabel("Frequency")
plt.hist(pd.DataFrame(X_scal)[4], bins = 50);

plt.subplot(4, 4, 6)
# гистограммa AveOccup     
plt.xlabel("AveOccup")
plt.ylabel("Frequency")
plt.hist(pd.DataFrame(X_scal)[5], bins = 50);

plt.subplot(4, 4, 7)
# гистограммa Latitude
plt.xlabel("Latitude")
plt.ylabel("Frequency")
plt.hist(pd.DataFrame(X_scal)[6], bins = 50);

plt.subplot(4, 4, 8)
# гистограммa Longitude    
plt.xlabel("Longitude")
plt.ylabel("Frequency")
plt.hist(pd.DataFrame(X_scal)[7], bins = 50);

plt.show();

X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X_scal), california_housing.target)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

# Создать эксперимент с названием
exp_name = 'sergey_zolotukhin'
exp_id = mlflow.create_experiment(name=exp_name) # ВНИМАНИЕ при повторном запуске, закомментировать строку!
mlflow.set_experiment(exp_name)

models = dict(zip(["HistGB", "LinearRegression", "LGBMRegressor"],
                  [HistGradientBoostingRegressor(), LinearRegression(), LGBMRegressor()]))

# Создадим parent run
with mlflow.start_run(run_name = "@Zolotukhin_S", experiment_id=mlflow.set_experiment(exp_name).experiment_id, description = "parent") as parent_run:
    for model_name in models.keys():
        with mlflow.start_run(run_name = model_name, experiment_id=mlflow.set_experiment(exp_name).experiment_id, nested = True) as child_run:
            model = models[model_name]
            
            # Обучим модель
            model.fit(pd.DataFrame(X_train), y_train)
            
            # Сделаем предсказание
            prediction = model.predict(X_val)
        
            # Создадим валидационный датасет
            eval_df = X_val.copy()
            eval_df["target"] = y_val
            # eval_df["prediction"] = prediction
        
            # Сохраним результаты обучения с помощью MLFlow
            signature = infer_signature(X_test, prediction)
            model_info = mlflow.sklearn.log_model(model, "log_reg", signature=signature,
                                                 registered_model_name=f"sk-learn-{model_name}-reg-model")
            mlflow.evaluate(
                model = model_info.model_uri,
                data = eval_df,
                targets = "target",
                # predictions = "prediction",
                model_type = "regressor",
                
            )  
