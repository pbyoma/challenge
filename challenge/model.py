
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
from typing import Tuple, Union, List

# Data Preprocessing Functions

def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()
    
    if morning_min <= date_time <= morning_max:
        return 'morning'
    elif afternoon_min <= date_time <= afternoon_max:
        return 'afternoon'
    else:
        return 'night'

def is_high_season(fecha):
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)
    
    if ((range1_min <= fecha <= range1_max) or 
        (range2_min <= fecha <= range2_max) or 
        (range3_min <= fecha <= range3_max) or
        (range4_min <= fecha <= range4_max)):
        return 1
    else:
        return 0

def get_min_diff(row):
    fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    return ((fecha_o - fecha_i).total_seconds()) / 60


# Data Preparation

data = pd.read_csv('../data/data.csv')

data['period_day'] = data['Fecha-I'].apply(get_period_day)
data['high_season'] = data['Fecha-I'].apply(is_high_season)
data['min_diff'] = data.apply(get_min_diff, axis=1)
threshold_in_minutes = 15
data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

# Model Training

# Feature Selection and Dummification
features = pd.concat([
    pd.get_dummies(data['OPERA'], prefix='OPERA'),
    pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
    pd.get_dummies(data['MES'], prefix='MES')
], axis=1)
target = data['delay']

# Data Split
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

# Training XGBoost Classifier
xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
xgb_model.fit(x_train, y_train)

# Training Logistic Regression
reg_model = LogisticRegression()
reg_model.fit(x_train, y_train)


class DelayModel:

    def __init__(self):
        self._model = xgb_model
        self.top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        # Generate period_day feature
        data['period_day'] = data['Fecha-I'].apply(get_period_day)
        # Generate high_season feature
        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        # Generate min_diff feature
        data['min_diff'] = data.apply(get_min_diff, axis=1)

        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        
        # Dummify categorical columns
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)

        features = features[self.top_10_features]
        
        if target_column:
            target = data[target_column]
            return features, target
        return features


    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.
        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Address class imbalance
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0 / n_y1
        
        # Train the XGBoost model
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.
        Args:
            features (pd.DataFrame): preprocessed data.
        Returns:
            (List[int]): predicted targets.
        """

        preds = self._model.predict(features)
        return [1 if pred > 0.5 else 0 for pred in preds]