from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, date_column='dteday', weekday_column='weekday'):
        self.date_column = date_column
        self.weekday_column = weekday_column


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Find the indices of NaN entries in the weekday column
        nan_indices = X[X[self.weekday_column].isnull()].index

        # Extract day names from the date column for these indices
        for idx in nan_indices:
            date_str = X.at[idx, self.date_column]
            date_object = datetime.strptime(date_str, '%Y-%m-%d')
            day_name = date_object.strftime('%A')

            # Impute the missing value with the day name
            X.at[idx, self.weekday_column] = day_name[:3]

        return X


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, column='weathersit'):
        self.column = column
        self.most_frequent = None

    def fit(self, X, y=None):
        print(self.column)
        self.most_frequent = X[self.column].mode()[0]
        return self

    def transform(self, X):
        # Fill missing values with the most frequent category
        X[self.column] = X[self.column].fillna(self.most_frequent)
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self):
        # Define mappings for each column
        self.mappings = {
            'yr': {2011: 0, 2012: 1},
            'mnth': {i: i for i in range(1, 13)},
            'season': {'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4},
            'weathersit': {'Clear': 1, 'Mist': 2, 'Light Rain': 3, 'Heavy Rain': 4},
            'holiday': {'No': 0, 'Yes': 1},
            'workingday': {'No': 0, 'Yes': 1},
            'hr': {
                '12am': 0, '1am': 1, '2am': 2, '3am': 3, '4am': 4, '5am': 5,
                '6am': 6, '7am': 7, '8am': 8, '9am': 9, '10am': 10, '11am': 11,
                '12pm': 12, '1pm': 13, '2pm': 14, '3pm': 15, '4pm': 16, '5pm': 17,
                '6pm': 18, '7pm': 19, '8pm': 20, '9pm': 21, '10pm': 22, '11pm': 23
            }
        }

    def fit(self, X, y=None):
        # No fitting necessary, mappings are predefined
        return self

    def transform(self, X):
        # Apply mappings to the specified columns
        X_transformed = X.copy()
        for column, mapping in self.mappings.items():
            if column in X_transformed.columns:
                X_transformed[column] = X_transformed[column].map(mapping)
        return X_transformed


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, factor=1.5):
        """
        Initialize the handler with optional columns and a factor for IQR.

        :param columns: List of columns to apply the outlier handling. If None, apply to all numerical columns.
        :param factor: The factor to multiply with IQR to determine bounds. Default is 1.5.
        """
        self.columns = columns
        self.factor = factor
        self.bounds = {}

    def fit(self, X, y=None):
        # Determine which columns to process
        columns_to_process = self.columns if self.columns is not None else X.select_dtypes(include=[np.number]).columns

        # Calculate bounds for each column
        for column in columns_to_process:
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            self.bounds[column] = {'lower': lower_bound, 'upper': upper_bound}

        return self

    def transform(self, X):
        X_transformed = X.copy()

        for column, bounds in self.bounds.items():
            lower_bound = bounds['lower']
            upper_bound = bounds['upper']

            X_transformed[column] = np.where(
                X_transformed[column] < lower_bound,
                lower_bound,
                X_transformed[column]
            )
            X_transformed[column] = np.where(
                X_transformed[column] > upper_bound,
                upper_bound,
                X_transformed[column]
            )

        return X_transformed

class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encode a weekday column.
    """

    def __init__(self, column=None):
        """
        Initialize the encoder with the column to be one-hot encoded.

        :param column: The name of the column containing weekday information.
        """
        self.column = column
        self.weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def fit(self, X, y=None):
        # Ensure the column exists in the DataFrame
        if self.column is None or self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' must be specified and exist in the DataFrame.")
        return self

    def transform(self, X):
        # Check if the column exists
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' does not exist in the DataFrame.")
        # make sure it is string type
        X[self.column] = X[self.column].astype(str)
        # Convert the weekday column to one-hot encoded columns
        X_transformed = X.copy()
        # cast the column to a categorical type
        X_transformed[self.column] = pd.Categorical(X_transformed[self.column], categories=self.weekdays)
        # the one hot encoding
        one_hot_encoded = pd.get_dummies(X_transformed[self.column], prefix=self.column)

        # Drop the original column and concatenate the new one-hot encoded columns
        X_transformed = X_transformed.drop(columns=[self.column])
        X_transformed = pd.concat([X_transformed, one_hot_encoded], axis=1)

        return X_transformed

class DropColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop the specified column
        if self.column_name in X.columns:
            return X.drop(self.column_name, axis=1)
        else:
            return X