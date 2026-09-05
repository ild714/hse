"""Полный воспроизводимый прогон исправленного анализа.

Запуск:  python run_analysis.py
Результат: таблицы в results/ и графики в results/figures/.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from src.data_prep import load_dataset, nominal_columns, numeric_columns
from src.factor import (
    adequacy,
    dominant_items,
    factor_block,
    fit_factors,
    mca,
    parallel_analysis,
)
from src.modeling import (
    build_models,
    evaluate,
    permutation_ranking,
    split,
    standardised_coefficients,
    test_report,
)

DATA = pathlib.Path(__file__).parent / "Спектакль_utf8_2.xlsx"
RESULTS = pathlib.Path(__file__).parent / "results"
FIGURES = RESULTS / "figures"


def save(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    RESULTS.mkdir(exist_ok=True)
    frame.to_csv(RESULTS / f"{name}.csv", index=True)
    return frame


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    features, target, score = load_dataset(str(DATA))

    section("1. ДАННЫЕ")
    print(f"наблюдений: {len(features)} (все строки сохранены)")
    print(f"признаков: {features.shape[1]} "
          f"({len(numeric_columns(features))} числовых, "
          f"{len(nominal_columns(features))} номинальных)")
    print("\nраспределение GGlevel:")
    print(target.value_counts().sort_index().to_string())

    # --- Моделирование -----------------------------------------------------
    x_train, x_test, y_train, y_test = split(features, target)
    models = build_models(features)

    section("2. КРОСС-ВАЛИДАЦИЯ НА ОБУЧАЮЩЕЙ ЧАСТИ")
    cv_table = evaluate(models, x_train, y_train)
    print(cv_table.round(4).to_string(index=False))
    save(cv_table, "cross_validation")

    section("3. ОТЛОЖЕННАЯ ВЫБОРКА (одинаковые метрики для всех моделей)")
    reports = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        reports[name] = test_report(model, x_test, y_test)
    test_table = pd.DataFrame(reports).T
    print(test_table.round(4).to_string())
    save(test_table, "test_metrics")

    best = models["logreg"]

    section("4. ВАЖНОСТЬ ПРИЗНАКОВ (перестановки на отложенной выборке)")
    importance = permutation_ranking(best, x_test, y_test)
    print(importance.head(20).round(4).to_string(index=False))
    save(importance, "permutation_importance")

    top = importance.head(15).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top["признак"], top["падение_f1"], xerr=top["std"], color="#4C72B0")
    plt.xlabel("падение macro-F1 при перестановке признака")
    plt.title("Важность признаков (permutation importance)")
    plt.tight_layout()
    plt.savefig(FIGURES / "permutation_importance.png", dpi=150)
    plt.close()

    coefficients = standardised_coefficients(best)
    save(coefficients, "logreg_coefficients")

    # --- Факторный анализ числового блока ----------------------------------
    section("5. ФАКТОРНЫЙ АНАЛИЗ ЧИСЛОВОГО БЛОКА")
    behaviour_columns = [
        c for c in numeric_columns(features) if not c.endswith("_is_missing")
    ]
    block = factor_block(features, behaviour_columns)
    checks = adequacy(block)
    for key, value in checks.items():
        print(f"  {key}: {value}")
    if checks["kmo"] < 0.6:
        print("\n  ВНИМАНИЕ: KMO ниже 0.6 — общей факторной структуры")
        print("  в поведенческом блоке практически нет.")

    horn = parallel_analysis(block, n_iter=200)
    n_factors = int(horn["оставляем"].sum())
    print(f"\n  параллельный анализ Хорна: {n_factors} факторов")
    print(f"  критерий Кайзера дал бы: {int((horn['собственное_значение'] > 1).sum())}")
    save(horn, "parallel_analysis")

    plt.figure(figsize=(8, 5))
    plt.plot(horn["фактор"], horn["собственное_значение"], "o-", label="наблюдаемые")
    plt.plot(horn["фактор"], horn["порог_случайности"], "s--",
             label="порог случайных данных (95-й перцентиль)")
    plt.axhline(1, color="gray", lw=1, ls=":", label="критерий Кайзера")
    plt.xlabel("номер фактора")
    plt.ylabel("собственное значение")
    plt.title("Параллельный анализ Хорна")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "parallel_analysis.png", dpi=150)
    plt.close()

    loadings, variance = fit_factors(block, n_factors, rotation="promax")
    print("\n  объяснённая дисперсия:")
    print(variance.round(3).to_string())
    print("\n  состав факторов (|нагрузка| >= 0.4):")
    for factor, items in dominant_items(loadings, 0.4).items():
        print(f"    {factor}: {'; '.join(items) if items else 'нет значимых нагрузок'}")
    save(loadings, "factor_loadings")

    plt.figure(figsize=(9, 8))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, linecolor="gray")
    plt.title("Факторные нагрузки числового блока (со знаком)")
    plt.tight_layout()
    plt.savefig(FIGURES / "factor_loadings.png", dpi=150)
    plt.close()

    # --- MCA номинального блока -------------------------------------------
    section("6. АНАЛИЗ СООТВЕТСТВИЙ (MCA) НОМИНАЛЬНОГО БЛОКА")
    summary, category_coords, respondent_coords = mca(
        features[nominal_columns(features)], n_components=6
    )
    print(summary.round(4).to_string(index=False))

    print("\n  связь осей с внешним критерием (rho Спирмена):")
    links = []
    for axis in respondent_coords.columns:
        rho_level = spearmanr(respondent_coords[axis], target).statistic
        rho_score = spearmanr(respondent_coords[axis], score).statistic
        links.append({"ось": axis, "rho_GGlevel": rho_level,
                      "rho_GGfscore": rho_score})
        print(f"    {axis}: GGlevel {rho_level:+.3f} | GGfscore {rho_score:+.3f}")
    links_table = pd.DataFrame(links)
    save(links_table, "mca_axes_vs_target")
    save(category_coords, "mca_category_coordinates")

    # Ось с наибольшей по модулю связью — содержательная ось компетентности.
    main_axis = links_table.loc[links_table["rho_GGfscore"].abs().idxmax(), "ось"]
    sign = np.sign(links_table.set_index("ось").loc[main_axis, "rho_GGfscore"])
    # Знак осей в SVD произволен; разворачиваем так, чтобы «больше» значило
    # «компетентнее» — иначе интерпретация полюсов читается наоборот.
    oriented = category_coords[main_axis] * sign
    print(f"\n  содержательная ось: {main_axis} "
          f"(rho с GGfscore = "
          f"{links_table.set_index('ось').loc[main_axis, 'rho_GGfscore']:+.3f})")
    print("\n  полюс НИЗКОЙ компетентности:")
    for name, value in oriented.sort_values().head(10).items():
        print(f"    {value:+.2f}  {name}")
    print("\n  полюс ВЫСОКОЙ компетентности:")
    for name, value in oriented.sort_values().tail(10).items():
        print(f"    {value:+.2f}  {name}")

    plt.figure(figsize=(8, 5))
    frame = pd.DataFrame({"ось": respondent_coords[main_axis] * sign,
                          "GGlevel": target})
    sns.boxplot(data=frame, x="GGlevel", y="ось", hue="GGlevel",
                palette="Blues", legend=False)
    plt.ylabel(f"{main_axis} (развёрнута: больше = компетентнее)")
    plt.title("Содержательная ось MCA против уровня цифровой грамотности")
    plt.tight_layout()
    plt.savefig(FIGURES / "mca_axis_vs_level.png", dpi=150)
    plt.close()

    print(f"\nГотово. Таблицы: {RESULTS}, графики: {FIGURES}")


if __name__ == "__main__":
    main()
