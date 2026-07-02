from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from urbanflow.modeling.feature_matrix import (
    DEFAULT_RIDGE_FEATURE_SPEC,
    ModelFeatureSpec,
    ModelTrainingError,
    select_model_features,
)


@dataclass(frozen=True)
class RidgeModelConfig:
    alpha: float = 1.0
    feature_spec: ModelFeatureSpec = DEFAULT_RIDGE_FEATURE_SPEC
    prediction_column: str = "ridge_prediction"


DEFAULT_RIDGE_MODEL_CONFIG = RidgeModelConfig()


@dataclass(frozen=True)
class FittedRidgeModel:
    pipeline: Pipeline
    config: RidgeModelConfig
    feature_columns: tuple[str, ...]
    training_row_count: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features, _ = select_model_features(
            frame,
            feature_spec=self.config.feature_spec,
            require_target=False,
        )
        predictions = self.pipeline.predict(features)
        return np.asarray(predictions, dtype=float)


def _build_ridge_pipeline(config: RidgeModelConfig) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(config.feature_spec.numeric_columns)),
            ("categorical", categorical_pipeline, list(config.feature_spec.categorical_columns)),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("features", preprocessor),
            ("model", Ridge(alpha=config.alpha)),
        ]
    )


def fit_ridge_model(
    train_frame: pd.DataFrame,
    *,
    config: RidgeModelConfig = DEFAULT_RIDGE_MODEL_CONFIG,
) -> FittedRidgeModel:
    features, target = select_model_features(train_frame, feature_spec=config.feature_spec)
    if target is None:
        raise ModelTrainingError("target is required for Ridge training")

    pipeline = _build_ridge_pipeline(config)
    pipeline.fit(features, target)
    return FittedRidgeModel(
        pipeline=pipeline,
        config=config,
        feature_columns=config.feature_spec.feature_columns,
        training_row_count=len(features),
    )


def add_ridge_predictions(frame: pd.DataFrame, fitted_model: FittedRidgeModel) -> pd.DataFrame:
    result = frame.copy()
    predictions = fitted_model.predict(result)
    result[fitted_model.config.prediction_column] = predictions
    return result
