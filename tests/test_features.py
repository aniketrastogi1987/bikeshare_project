"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeathersitImputer


def test_weathersit_transformer(sample_input_data):
    # Given
    transformer = WeathersitImputer(
        column=config.model_config_.weathersit_var  # Use 'column' instead of 'columns'
    )

    # Print the indices of the DataFrame
    # print(f"Indices of the DataFrame: {sample_input_data[0].index}")

    # Find a valid index where 'weathersit' is null
    valid_index = sample_input_data[0][sample_input_data[0]['weathersit'].isnull()].index[0]
    print(f"Value before transformation: {sample_input_data[0].loc[valid_index, 'weathersit']}")
    print(f"Row before transformation: {sample_input_data[0].loc[valid_index]}")
    assert pd.isnull(sample_input_data[0].loc[valid_index, 'weathersit'])  # Use pd.isnull instead of np.isnan
    print(f"Most frequent value: {sample_input_data[0]}")

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Debug: Print the actual value
    actual_value = subject.loc[valid_index, 'weathersit']
    expected_value = transformer.most_frequent
    print(f"Actual value after transformation: {actual_value}")
    print(f"Row after transformation: {subject.loc[valid_index]}")
    print(f"Expected most frequent value: {expected_value}")

    # Then
    assert actual_value == expected_value  # Assert the most frequent value