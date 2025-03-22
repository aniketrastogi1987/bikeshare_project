import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import WeekdayOneHotEncoder
from bikeshare_model.processing.features import DropColumn
#from bikeshare_model.processing.data_manager import pre_pipeline_preparation

# Assuming pre_pipeline_preparation returns a dictionary or similar structure
#data_prep = pre_pipeline_preparation(data_frame=data)
#numeric_cols = data_prep['numeric_cols']

pipeline = Pipeline([
    ('weekday_imputer', WeekdayImputer()),
    ('weathersit_imputer', WeathersitImputer()),
    ('mapper', Mapper()),
    ('outlier_handler', OutlierHandler(columns=['temp', 'atemp', 'hum', 'windspeed'])),
    ('weekday_encoder', WeekdayOneHotEncoder(column='weekday')),
    ('drop_column', DropColumn(column_name='dteday')), # Drop the column here
    ('regressor', RandomForestRegressor()) # You can change the regressor here
])