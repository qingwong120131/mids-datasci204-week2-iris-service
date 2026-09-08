import pytest
from iris_service.features import validate_measurements, to_model_input, InvalidFeatureError

VALID = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


def test_valid_measurements_pass():
    validate_measurements(VALID)  # should not raise


def test_missing_feature_raises():
    with pytest.raises(InvalidFeatureError):
        validate_measurements({"sepal_length": 5.1})


def test_negative_value_raises():
    bad = {**VALID, "sepal_length": -1}
    with pytest.raises(InvalidFeatureError):
        validate_measurements(bad)


def test_zero_value_raises():
    bad = {**VALID, "sepal_length": 0}
    with pytest.raises(InvalidFeatureError):
        validate_measurements(bad)


def test_out_of_range_raises():
    bad = {**VALID, "sepal_length": 500}
    with pytest.raises(InvalidFeatureError):
        validate_measurements(bad)


def test_to_model_input_orders_correctly():
    scrambled = {"petal_width": 0.2, "petal_length": 1.4, "sepal_width": 3.5, "sepal_length": 5.1}
    assert to_model_input(scrambled) == [5.1, 3.5, 1.4, 0.2]