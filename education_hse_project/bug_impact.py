"""Количественная оценка вклада главной ошибки исходных ноутбуков.

В исходных ноутбуках (ячейки 20, 21, 39 во всех трёх версиях) стояло:

    data.loc[data['fulltime'] == 'undefined'] = 0
    data.loc[data['firstclick'] == 'undefined'] = 0

Здесь пропущено указание столбца. ``data.loc[mask] = 0`` обнуляет строку
целиком — вместе с целевой переменной GGlevel. Скрипт показывает, к чему
это приводит: сколько строк испорчено и насколько завышается качество
модели на испорченных данных.

Запуск:  python bug_impact.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from src.data_prep import (
    RAW_PREFIX,
    build_features,
    load_raw,
)
from src.modeling import build_models, split, test_report

DATA = pathlib.Path(__file__).parent / "Спектакль_utf8_2.xlsx"


def reproduce_original_bug(raw: pd.DataFrame) -> pd.DataFrame:
    """Воспроизводит обнуление строк ровно так, как в исходных ноутбуках."""
    # pandas 3 уже запрещает записывать int в строковый столбец, поэтому
    # для точного воспроизведения поведения старых версий приводим кадр
    # к object — иначе ошибка просто не выполнится.
    data = raw.astype(object)
    data.loc[data["fulltime"] == "undefined"] = 0
    data.loc[data["fulltime"] == "nun"] = 0
    data.loc[data["firstclick"] == "undefined"] = 0
    return data


def main() -> None:
    raw = load_raw(str(DATA))

    affected = (
        (raw["fulltime"] == "undefined")
        | (raw["fulltime"] == "nun")
        | (raw["firstclick"] == "undefined")
    )
    true_levels = raw.loc[affected, "GGlevel"].value_counts().sort_index()

    print("=" * 70)
    print("ВЛИЯНИЕ ОШИБКИ data.loc[mask] = 0")
    print("=" * 70)
    print(f"строк в выгрузке:        {len(raw)}")
    print(f"строк обнулено целиком:  {int(affected.sum())} "
          f"({affected.mean():.1%})")
    print("\nих настоящий GGlevel до обнуления:")
    print(true_levels.to_string())
    print(f"\nиз них имели GGlevel != 0: {int(true_levels.drop(0).sum())} "
          "— этим респондентам целевая переменная была переписана на 0")

    corrupted = reproduce_original_bug(raw)
    print(f"\nпосле обнуления доля GGlevel = 0 выросла с "
          f"{(raw['GGlevel'] == 0).mean():.1%} до "
          f"{(corrupted['GGlevel'] == 0).mean():.1%}")

    duplicates = corrupted.loc[affected].drop_duplicates()
    print(f"обнулённые строки схлопнулись в {len(duplicates)} уникальную строку "
          "из одних нулей — модель отделяет их тривиально")

    print("\n" + "=" * 70)
    print("КАЧЕСТВО МОДЕЛИ: испорченные данные против исправленных")
    print("=" * 70)

    rows = []
    for label, frame in [("испорченные (как в оригинале)", corrupted),
                         ("исправленные", raw)]:
        features, target, _ = build_features(frame)
        x_train, x_test, y_train, y_test = split(features, target)
        model = build_models(features)["logreg"]
        model.fit(x_train, y_train)
        metrics = test_report(model, x_test, y_test)
        metrics["данные"] = label
        rows.append(metrics)

    table = pd.DataFrame(rows).set_index("данные")
    print(table.round(4).to_string())

    inflation = table.loc["испорченные (как в оригинале)", "accuracy"] - table.loc[
        "исправленные", "accuracy"
    ]
    print(f"\nзавышение accuracy за счёт одной этой ошибки: {inflation:+.4f}")
    print("Ошибка создаёт искусственный сигнал: 388 одинаковых строк из нулей,")
    print("все помеченные уровнем 0. Модель «угадывает» их безошибочно,")
    print("и метрика растёт, хотя предсказательная способность не улучшилась.")


if __name__ == "__main__":
    main()
