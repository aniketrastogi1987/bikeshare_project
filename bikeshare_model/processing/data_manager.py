import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import re
import joblib
import pandas as pd
import typing as t
from sklearn.pipeline import Pipeline
import datetime
import numpy as np

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


##  Pre-Pipeline Preparation

def extract_year_month(df):
    year = df['dteday'].apply(lambda x: int(x.split('-')[0]))
    month = df['dteday'].apply(lambda x: int(x.split('-')[1]))
    return year, month
""""
def convert_to_24_hour(time_str):
    if 'am' in time_str:
        hour = int(time_str.replace('am', '').strip())
        return hour if hour != 12 else 0  # Convert 12am to 0
    elif 'pm' in time_str:
        hour = int(time_str.replace('pm', '').strip())
        return hour + 12 if hour != 12 else 12
"""

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    data_frame['year'] = extract_year_month(data_frame)[0]
    data_frame['month'] = extract_year_month(data_frame)[1]
#    data_frame['hr'] = data_frame['hr'].apply(convert_to_24_hour)
    
    numeric_cols = data_frame.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data_frame.select_dtypes(exclude=np.number).columns.tolist()

    #data_frame['Has_cabin']=data_frame['Cabin'].apply(f1)               #  processing cabin 

    # drop unnecessary variables
    unused_fields = [field for field in config.model_config_.unused_fields if field in data_frame.columns]
    data_frame.drop(labels=unused_fields, axis=1, inplace=True)
    
    return data_frame



def load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)
    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    print("Model/pipeline trained successfully!")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py", ".gitignore"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

