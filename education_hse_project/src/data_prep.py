"""Загрузка, очистка и конструирование признаков для логов задания «Спектакль».

Модуль сознательно отделён от моделирования: все решения о том, что считать
пропуском и как превращать сырой лог в признак, собраны здесь в одном месте.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

RANDOM_STATE = 42

# В выгрузке пропуск закодирован тремя разными способами: пустая ячейка,
# строка 'undefined' и опечатка 'nun'. Все три означают «действие не
# зафиксировано» и должны стать NaN.
MISSING_TOKENS = ["undefined", "nun", "NaN", ""]

RAW_PREFIX = "1TLDCDSILC_log_"

# Целевая переменная и её непрерывный источник. GGfscore — это тот же
# конструкт, из которого нарезан GGlevel, поэтому в признаки он попасть
# не может: это прямая утечка целевой переменной.
TARGET = "GGlevel"
TARGET_CONTINUOUS = "GGfscore"
LEAKY = [TARGET_CONTINUOUS, "Username"]

# Числовые признаки: время и счётчики событий.
NUMERIC = [
    "fulltime",
    "openglasha",
    "notion",
    "savepictureN",
    "picturechooseN",
    "authorN",
    "authortime",
]

# Номинальные признаки: выбор из фиксированного набора вариантов.
# Порядка между вариантами нет, поэтому кодируются one-hot, а не ordinal.
NOMINAL = [
    "exit",
    "firstclick",
    "loginchoose",
    "help",
    "commtypes",
    "commname",
    "commеtheme",  # в исходной выгрузке буква «е» русская — сохраняем как есть
    "commdescription",
    "commfoto",
    "privacy",
    "mail1",
    "mail2",
    "wall1",
    "private1",
    "private2",
    "wall2",
    "wall2comment3",
    "wall2comment4",
    "wall2comment5",
    "private3",
    "private4",
    "private5",
    "private6",
    "wall3",
    "wall4",
]

# Столбцы-последовательности: лог записывал сюда порядок кликов через запятую.
# Excel в русской локали часть таких значений превратил в дробные числа
# («2,4» -> 2.4), поэтому парсер обязан понимать оба написания.
SEQUENCE = ["savepicture", "picturechoose", "picturechoosedelete", "citationcode"]

# Свободный текст, скопированный респондентом. Правильный ответ задания —
# найти рабочую почту Марии Соловьёвой.
CORRECT_EMAIL = "soloveym@teatr.ru"


def load_raw(path: str) -> pd.DataFrame:
    """Читает выгрузку, приводит имена столбцов и убирает служебные колонки."""
    data = pd.read_excel(path, index_col=0)
    data.columns = [c.replace(RAW_PREFIX, "") for c in data.columns]
    junk = [c for c in data.columns if c.startswith("Unnamed")]
    return data.drop(columns=junk)


def normalise_missing(data: pd.DataFrame) -> pd.DataFrame:
    """Сводит все кодировки пропуска к NaN.

    Именно здесь исправлена ошибка исходных ноутбуков: там стояло
    ``data.loc[data['fulltime'] == 'undefined'] = 0``, что обнуляло строку
    целиком — вместе с целевой переменной GGlevel.
    """
    out = data.copy()
    for column in out.columns:
        out[column] = out[column].replace(MISSING_TOKENS, np.nan)
    return out


def parse_sequence(value) -> list[int]:
    """Разбирает значение столбца-последовательности в список кодов действий.

    Excel исказил эти данные тремя разными способами, и парсер обязан
    понимать все формы записи, встречающиеся в выгрузке:

    * ``'2,4,5'`` — уцелевшая исходная строка;
    * ``2.4`` и ``'5.0999999999999996'`` — пара «2,4», прочитанная как дробь
      (запятая = десятичный разделитель в русской локали), возможно с шумом
      представления float;
    * ``Timestamp('2024-01-02')`` — пара «2,1», прочитанная как дата
      «2 января»; день и месяц у всех таких значений лежат в 1..8, то есть
      ровно в алфавите кодов действий;
    * ``2`` — одно действие.
    """
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (datetime.datetime, pd.Timestamp)):
        return [value.day, value.month]
    if isinstance(value, (int, np.integer)):
        return [int(value)]

    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            return [int(p) for p in text.split(",") if p.strip().isdigit()]
        try:
            value = float(text)
        except ValueError:
            return []

    number = float(value)
    if number.is_integer():
        return [int(number)]
    # Дробная часть кодирует ровно один следующий код действия, поэтому
    # округляем до одного знака и тем самым снимаем шум представления float.
    whole, _, frac = f"{number:.1f}".partition(".")
    return [int(whole), int(frac)]


def sequence_features(data: pd.DataFrame) -> pd.DataFrame:
    """Числовые признаки последовательностей кликов.

    Сам код последовательности («2,4,5») — идентификатор, а не число:
    сравнивать его с «1,3» бессмысленно. Зато информативны её длина,
    разнообразие и наличие повторов — это и есть паттерн работы.
    """
    out = pd.DataFrame(index=data.index)
    for column in SEQUENCE:
        parsed = data[column].map(parse_sequence)
        # Хранятся ровно две из трёх величин: число уникальных объектов
        # равно длине минус число повторов, и добавлять его третьим
        # означало бы внести точную линейную зависимость — она делает
        # корреляционную матрицу вырожденной и ломает факторный анализ.
        out[f"{column}_n_actions"] = parsed.map(len)
        out[f"{column}_n_repeats"] = parsed.map(lambda s: len(s) - len(set(s)))
    return out


def sequence_nominal(data: pd.DataFrame) -> pd.DataFrame:
    """Номинальные признаки последовательностей: какой объект выбран первым.

    Номер картинки — это метка, а не величина, поэтому такой признак идёт
    в one-hot вместе с остальными категориальными, а не в числовые.
    """
    out = pd.DataFrame(index=data.index)
    for column in SEQUENCE:
        parsed = data[column].map(parse_sequence)
        out[f"{column}_first"] = parsed.map(
            lambda s: str(s[0]) if s else "__MISSING__"
        )
    return out


def citation_features(data: pd.DataFrame) -> pd.DataFrame:
    """Признаки по списку источников, которыми пользовался респондент."""
    out = pd.DataFrame(index=data.index)
    text = data["citation"].fillna("")
    out["citation_n_sources"] = text.map(lambda s: len(s.split()) if s else 0)
    return out


def copymail_features(data: pd.DataFrame) -> pd.DataFrame:
    """Признаки по скопированному тексту с контактами.

    Задание проверяет, сумеет ли респондент вычленить именно рабочий адрес.
    Поэтому важно не само содержимое, а три вещи: найден ли верный адрес,
    скопирован ли он «чисто» и сколько лишнего текста прихвачено.
    """
    out = pd.DataFrame(index=data.index)
    text = data["copymail"].fillna("").astype(str)
    stripped = text.str.strip()
    out["copymail_has_correct_email"] = stripped.str.contains(
        CORRECT_EMAIL, regex=False
    ).astype(int)
    out["copymail_is_exact_email"] = (stripped == CORRECT_EMAIL).astype(int)
    out["copymail_len"] = stripped.str.len()
    out["copymail_is_empty"] = (stripped == "").astype(int)
    return out


def build_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Собирает матрицу признаков и целевую переменную.

    Возвращает (признаки, GGlevel, GGfscore). Ни одна строка не удаляется:
    пропуск здесь информативен сам по себе, а listwise deletion выбрасывала
    бы в первую очередь слабых респондентов и смещала выборку.
    """
    clean = normalise_missing(data)

    target = clean[TARGET].astype(int)
    target_score = clean[TARGET_CONTINUOUS].astype(float)

    numeric = clean[NUMERIC].apply(pd.to_numeric, errors="coerce")
    # fulltime записан в секундах — переводим в минуты без потери дробной части.
    numeric["fulltime"] = numeric["fulltime"] / 60.0

    nominal = clean[NOMINAL].astype("object")
    # Явная категория для пропуска: «не сделал» — это тоже поведение.
    nominal = nominal.where(nominal.notna(), "__MISSING__").astype(str)
    nominal = pd.concat([nominal, sequence_nominal(clean)], axis=1)

    engineered = pd.concat(
        [
            sequence_features(clean),
            citation_features(clean),
            copymail_features(clean),
        ],
        axis=1,
    )

    # Индикаторы пропуска для числовых столбцов: у authorN, authortime и
    # picturechoosedelete пропущено больше 70 % значений, и сам факт пропуска
    # несёт сигнал.
    indicators = pd.DataFrame(index=clean.index)
    for column in NUMERIC:
        if numeric[column].isna().any():
            indicators[f"{column}_is_missing"] = numeric[column].isna().astype(int)

    features = pd.concat([numeric, engineered, indicators, nominal], axis=1)
    return features, target, target_score


def nominal_columns(features: pd.DataFrame) -> list[str]:
    """Столбцы, которые кодируются one-hot.

    Список задан явно, а не выведен из dtype: определение типа по dtype
    ломается между версиями pandas, а ошибка здесь тихо отправила бы
    категории в числовой блок — ровно та ошибка, которую мы исправляем.
    """
    expected = set(NOMINAL) | {f"{c}_first" for c in SEQUENCE}
    return [c for c in features.columns if c in expected]


def numeric_columns(features: pd.DataFrame) -> list[str]:
    """Столбцы, которые масштабируются как числа."""
    nominal = set(nominal_columns(features))
    return [c for c in features.columns if c not in nominal]


def load_dataset(path: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Полный путь от файла до готовых признаков."""
    return build_features(load_raw(path))
