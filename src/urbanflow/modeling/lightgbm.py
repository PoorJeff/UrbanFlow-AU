from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from urbanflow.modeling.feature_matrix import (
    DEFAULT_RIDGE_FEATURE_SPEC,
    ModelFeatureSpec,
    ModelTrainingError,
    select_model_features,
)


@dataclass(frozen=True)
class LightGBMModelConfig:
    n_estimators: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    random_state: int = 42
    feature_spec: ModelFeatureSpec = DEFAULT_RIDGE_FEATURE_SPEC
    prediction_column: str = "lightgbm_prediction"


DEFAULT_LIGHTGBM_MODEL_CONFIG = LightGBMModelConfig()


@dataclass(frozen=True)
class FittedLightGBMModel:
    pipeline: Pipeline
    config: LightGBMModelConfig
    feature_columns: tuple[str, ...]
    training_row_count: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features, _ = select_model_features(
            frame,
            feature_spec=self.config.feature_spec,
            require_target=False,
        )
        predictions = self.pipeline.predict(features)
        return _clip_nonnegative_predictions(predictions)


def _clip_nonnegative_predictions(predictions: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(predictions), dtype=float)
    return np.maximum(values, 0.0)


def _build_lightgbm_pipeline(config: LightGBMModelConfig) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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
    preprocessor.set_output(transform="pandas")
    model = LGBMRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        random_state=config.random_state,
        verbosity=-1,
    )
    return Pipeline(
        steps=[
            ("features", preprocessor),
            ("model", model),
        ]
    )


def fit_lightgbm_model(
    train_frame: pd.DataFrame,
    *,
    config: LightGBMModelConfig = DEFAULT_LIGHTGBM_MODEL_CONFIG,
) -> FittedLightGBMModel:
    features, target = select_model_features(train_frame, feature_spec=config.feature_spec)
    if target is None:
        raise ModelTrainingError("target is required for LightGBM training")

    pipeline = _build_lightgbm_pipeline(config)
    pipeline.fit(features, target)
    return FittedLightGBMModel(
        pipeline=pipeline,
        config=config,
        feature_columns=config.feature_spec.feature_columns,
        training_row_count=len(features),
    )


def add_lightgbm_predictions(
    frame: pd.DataFrame,
    fitted_model: FittedLightGBMModel,
) -> pd.DataFrame:
    result = frame.copy()
    predictions = fitted_model.predict(result)
    result[fitted_model.config.prediction_column] = predictions
    return result
