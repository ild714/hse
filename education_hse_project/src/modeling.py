"""Обучение и честная оценка моделей уровня цифровой грамотности.

Ключевое отличие от исходных ноутбуков: GGlevel рассматривается как то,
чем он и является — порядковая метка из четырёх уровней. Поэтому решается
задача классификации, а базовая модель сравнивается с обученной по одной
и той же метрике.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_prep import RANDOM_STATE, nominal_columns, numeric_columns


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Препроцессор, целиком лежащий внутри пайплайна.

    Импутация и масштабирование обучаются только на train-фолде, поэтому
    информация из теста не протекает в обучение.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", min_frequency=10)
    return ColumnTransformer(
        [
            ("num", numeric, numeric_columns(features)),
            ("cat", categorical, nominal_columns(features)),
        ]
    )


def build_models(features: pd.DataFrame) -> dict[str, Pipeline]:
    """Набор моделей: две базовые и три содержательные."""
    pre = lambda: make_preprocessor(features)  # noqa: E731
    return {
        "baseline_most_frequent": Pipeline(
            [("pre", pre()), ("clf", DummyClassifier(strategy="most_frequent"))]
        ),
        "baseline_stratified": Pipeline(
            [
                ("pre", pre()),
                (
                    "clf",
                    DummyClassifier(
                        strategy="stratified", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "logreg": Pipeline(
            [
                ("pre", pre()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        C=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("pre", pre()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("pre", pre()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.06,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def quadratic_kappa(y_true, y_pred) -> float:
    """Взвешенная каппа — метрика, учитывающая порядок уровней.

    Ошибка «спутал уровень 0 с уровнем 3» штрафуется сильнее, чем
    «спутал 1 с 2». Обычная accuracy этой разницы не видит.
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


SCORING = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
}


def evaluate(
    models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Кросс-валидация всех моделей на обучающей части."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model, x_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1
        )
        rows.append(
            {
                "model": name,
                "accuracy": scores["test_accuracy"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "f1_macro": scores["test_f1_macro"].mean(),
                "f1_macro_std": scores["test_f1_macro"].std(),
            }
        )
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False)


def test_report(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Все метрики на отложенной выборке — одним набором для любой модели."""
    pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "quadratic_kappa": quadratic_kappa(y_test, pred),
        "mae_levels": mean_absolute_error(y_test, pred),
    }


def split(features: pd.DataFrame, target: pd.Series):
    """Стратифицированное разбиение с фиксированным random_state.

    Отложенная выборка используется ровно один раз — в самом конце.
    Отбор модели идёт по кросс-валидации внутри обучающей части.
    """
    return train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )


def feature_names(model: Pipeline) -> list[str]:
    return list(model.named_steps["pre"].get_feature_names_out())


def standardised_coefficients(model: Pipeline) -> pd.DataFrame:
    """Коэффициенты логистической регрессии по классам.

    Признаки уже стандартизованы внутри пайплайна, поэтому величины
    коэффициентов сопоставимы между собой — в отличие от исходной работы,
    где время в минутах соседствовало с кодами ответов.
    """
    clf = model.named_steps["clf"]
    frame = pd.DataFrame(
        clf.coef_.T,
        index=feature_names(model),
        columns=[f"уровень_{c}" for c in clf.classes_],
    )
    frame["макс_модуль"] = frame.abs().max(axis=1)
    return frame.sort_values("макс_модуль", ascending=False)


def permutation_ranking(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Важность признаков через перестановки на отложенной выборке.

    В отличие от коэффициентов, эта оценка не зависит от вида модели
    и от того, как закодированы категории: она измеряет реальное падение
    качества при разрушении связи признака с целевой переменной.
    """
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "признак": x_test.columns,
                "падение_f1": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("падение_f1", ascending=False)
        .reset_index(drop=True)
    )
