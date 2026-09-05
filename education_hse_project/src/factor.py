"""Поиск латентной структуры в логах.

Исходная работа применяла классический факторный анализ ко всем столбцам
сразу, предварительно закодировав категории порядковыми номерами. Это
некорректно: факторный анализ строится на корреляциях Пирсона и требует
переменных, для которых расстояние осмысленно, а у номинальных категорий
(«почта» / «поиск») его нет.

Поэтому здесь два инструмента, каждый на своём типе данных:

* классический факторный анализ — на числовом блоке (время, счётчики,
  характеристики последовательностей), где предпосылки выполняются;
* множественный анализ соответствий (MCA) — на номинальном блоке,
  это прямой аналог факторного анализа для категориальных данных.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_bartlett_sphericity,
    calculate_kmo,
)

from .data_prep import RANDOM_STATE

# Столбцы, у которых пропущено больше половины значений: восстанавливать
# такую переменную медианой бессмысленно, в факторный анализ она не идёт.
MOSTLY_EMPTY = ["authorN", "authortime"]


def factor_block(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Готовит числовой блок к факторному анализу.

    Пропуски заполняются медианой, а не удалением строк: listwise deletion
    на этих данных оставила бы 462 наблюдения из 2796 и сместила бы выборку
    в сторону тех, кто дошёл до конца задания.
    """
    block = features[[c for c in columns if c not in MOSTLY_EMPTY]].copy()
    block = block.fillna(block.median())
    return block.loc[:, block.std() > 0]


def adequacy(block: pd.DataFrame) -> dict:
    """Проверка пригодности данных для факторного анализа.

    Дополнительно возвращает определитель корреляционной матрицы: если он
    близок к нулю, матрица вырождена и χ² Бартлетта считать нельзя. В
    исходной работе этот случай как раз и наступил — numpy печатал
    `divide by zero encountered in det`, но результат всё равно был
    истолкован как значимый.
    """
    data = block
    corr = np.corrcoef(data.values, rowvar=False)
    determinant = float(np.linalg.det(corr))
    chi2, p_value = calculate_bartlett_sphericity(data)
    _, kmo_model = calculate_kmo(data)
    return {
        "n_rows": len(data),
        "determinant": determinant,
        "bartlett_chi2": float(chi2),
        "bartlett_p": float(p_value),
        "kmo": float(kmo_model),
        "matrix_is_singular": abs(determinant) < 1e-12,
    }


def parallel_analysis(
    block: pd.DataFrame, n_iter: int = 200, percentile: int = 95
) -> pd.DataFrame:
    """Параллельный анализ Хорна — корректный способ выбрать число факторов.

    Критерий Кайзера («собственное значение больше 1») систематически
    завышает число факторов. Параллельный анализ сравнивает наблюдаемые
    собственные значения с теми, что возникают на случайных данных того же
    размера, и оставляет только превосходящие случайный уровень.
    """
    data = block
    observed = FactorAnalyzer(rotation=None).fit(data).get_eigenvalues()[0]

    rng = np.random.default_rng(RANDOM_STATE)
    simulated = np.empty((n_iter, data.shape[1]))
    for i in range(n_iter):
        noise = rng.normal(size=data.shape)
        simulated[i] = FactorAnalyzer(rotation=None).fit(noise).get_eigenvalues()[0]
    threshold = np.percentile(simulated, percentile, axis=0)

    return pd.DataFrame(
        {
            "фактор": np.arange(1, len(observed) + 1),
            "собственное_значение": observed,
            "порог_случайности": threshold,
            "оставляем": observed > threshold,
        }
    )


def fit_factors(
    block: pd.DataFrame, n_factors: int, rotation: str = "promax"
) -> pd.DataFrame:
    """Факторные нагрузки со знаком.

    Знак сохраняется намеренно: в исходной работе стояло
    ``loadings = np.abs(loadings)``, после чего направление связи было
    потеряно, а выводы о «повышающем влиянии» фактора уже не опирались
    на данные.
    """
    data = block
    model = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    model.fit(data)
    loadings = pd.DataFrame(
        model.loadings_,
        index=data.columns,
        columns=[f"Фактор{i + 1}" for i in range(n_factors)],
    )
    variance = pd.DataFrame(
        model.get_factor_variance(),
        index=["дисперсия", "доля_дисперсии", "накопленная_доля"],
        columns=loadings.columns,
    )
    return loadings, variance


def dominant_items(
    loadings: pd.DataFrame, threshold: float = 0.4
) -> dict[str, list[str]]:
    """Для каждого фактора — переменные, которые на него значимо грузятся."""
    groups = {}
    for factor in loadings.columns:
        strong = loadings[factor][loadings[factor].abs() >= threshold]
        groups[factor] = [
            f"{name} ({value:+.2f})"
            for name, value in strong.sort_values(key=abs, ascending=False).items()
        ]
    return groups


def mca(block: pd.DataFrame, n_components: int = 5):
    """Множественный анализ соответствий для номинальных признаков.

    Аналог факторного анализа, корректный для категорий: работает с
    таблицей индикаторов, не требует упорядоченности вариантов и не
    приписывает категориям числовых значений.

    Возвращает (собственные значения с поправкой Бенцекри, координаты
    категорий по осям, координаты респондентов по осям).
    """
    indicator = pd.get_dummies(block.astype(str)).astype(float)
    n_vars = block.shape[1]

    total = indicator.values.sum()
    correspondence = indicator.values / total
    row_mass = correspondence.sum(axis=1)
    col_mass = correspondence.sum(axis=0)

    expected = np.outer(row_mass, col_mass)
    residual = (correspondence - expected) / np.sqrt(expected)
    u, singular, vt = np.linalg.svd(residual, full_matrices=False)

    eigenvalues = singular[:n_components] ** 2

    # Поправка Бенцекри: сырые собственные значения MCA систематически
    # занижают долю объяснённой инерции, поэтому их принято корректировать.
    keep = eigenvalues > 1.0 / n_vars
    corrected = np.where(
        keep,
        ((n_vars / (n_vars - 1)) * (eigenvalues - 1.0 / n_vars)) ** 2,
        0.0,
    )
    explained = corrected / corrected.sum() if corrected.sum() > 0 else corrected

    axes = [f"Ось{i + 1}" for i in range(n_components)]
    coordinates = pd.DataFrame(
        (vt[:n_components].T / np.sqrt(col_mass)[:, None]) * singular[:n_components],
        index=indicator.columns,
        columns=axes,
    )
    # Координаты респондентов на тех же осях: позволяют проверить, связана ли
    # найденная структура с внешним критерием (уровнем цифровой грамотности).
    row_coordinates = pd.DataFrame(
        (u[:, :n_components] / np.sqrt(row_mass)[:, None]) * singular[:n_components],
        index=block.index,
        columns=axes,
    )
    summary = pd.DataFrame(
        {
            "ось": np.arange(1, n_components + 1),
            "собственное_значение": eigenvalues,
            "доля_инерции_бенцекри": explained,
        }
    )
    return summary, coordinates, row_coordinates
