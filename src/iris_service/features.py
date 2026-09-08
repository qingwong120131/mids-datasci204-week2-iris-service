"""Feature validation and shaping for the iris prediction service."""

FEATURE_ORDER = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


class InvalidFeatureError(ValueError):
    """Raised when input features fail validation."""


def validate_measurements(features: dict[str, float]) -> None:
    """Ensure all required measurements are present and physically plausible."""
    missing = [f for f in FEATURE_ORDER if f not in features]
    if missing:
        raise InvalidFeatureError(f"Missing required features: {missing}")

    for name in FEATURE_ORDER:
        value = features[name]
        if value <= 0:          # <-- seeded bug: should be `<= 0`, caught in Step 5
            raise InvalidFeatureError(f"{name} must be positive, got {value}")
        if value > 30:          # sanity bound in cm
            raise InvalidFeatureError(f"{name} exceeds plausible range, got {value}")


def to_model_input(features: dict[str, float]) -> list[float]:
    """Convert a validated feature dict into the ordered list the model expects."""
    validate_measurements(features)
    return [features[name] for name in FEATURE_ORDER]