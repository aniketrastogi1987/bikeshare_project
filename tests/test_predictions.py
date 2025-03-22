import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from bikeshare_model.predict import make_prediction
from bikeshare_model.config.core import config


def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = len(sample_input_data[0])

    # When
    result = make_prediction(input_data=sample_input_data[0])

    # Debug: Print the actual results
    print(f"Predictions: {result['predictions']}")
    print(f"Errors: {result['errors']}")

    # Then
    assert result["predictions"] is not None
    assert len(result["predictions"]) == expected_no_predictions
    assert result["errors"] is None