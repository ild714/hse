"""Тесты разбора данных.

Основное внимание — парсеру последовательностей: именно там Excel исказил
значения тремя разными способами, и молчаливая ошибка разбора не видна
глазом, но портит все последующие выводы.
"""

import datetime

import pandas as pd
import pytest

from src.data_prep import (
    MISSING_TOKENS,
    build_features,
    normalise_missing,
    nominal_columns,
    numeric_columns,
    parse_sequence,
)


class TestParseSequence:
    def test_single_integer(self):
        assert parse_sequence(4) == [4]

    def test_comma_separated_string(self):
        assert parse_sequence("1,5,3,2,4") == [1, 5, 3, 2, 4]

    def test_float_is_a_pair_mangled_by_excel(self):
        # «2,4» в русской локали Excel прочитал как дробь 2.4
        assert parse_sequence(2.4) == [2, 4]

    def test_float_noise_is_rounded_away(self):
        # Так 5.1 и 8.3 выглядят в выгрузке после сериализации float
        assert parse_sequence("5.0999999999999996") == [5, 1]
        assert parse_sequence("8.3000000000000007") == [8, 3]

    def test_integral_float_is_one_action(self):
        assert parse_sequence(4.0) == [4]

    def test_datetime_is_a_day_month_pair(self):
        # «2,1» Excel прочитал как 2 января
        assert parse_sequence(datetime.datetime(2024, 1, 2)) == [2, 1]
        assert parse_sequence(pd.Timestamp("2024-02-01")) == [1, 2]

    def test_missing_gives_empty_sequence(self):
        assert parse_sequence(None) == []
        assert parse_sequence(float("nan")) == []

    def test_repeats_are_preserved(self):
        # Повтор — содержательный признак, схлопывать его нельзя
        assert parse_sequence("2,2,2,2,2") == [2, 2, 2, 2, 2]

    @pytest.mark.parametrize("value", ["1,5,3,2,4", 2.4, 4, pd.Timestamp("2024-03-05")])
    def test_codes_stay_within_alphabet(self, value):
        assert all(1 <= code <= 8 for code in parse_sequence(value))


class TestNormaliseMissing:
    def test_all_missing_tokens_become_nan(self):
        frame = pd.DataFrame({"a": ["undefined", "nun", "почта"]})
        result = normalise_missing(frame)
        assert result["a"].isna().sum() == 2
        assert result["a"].iloc[2] == "почта"

    def test_target_is_never_touched(self):
        """Регрессия на главную ошибку исходных ноутбуков.

        Там ``data.loc[data['fulltime'] == 'undefined'] = 0`` обнуляло строку
        целиком, включая GGlevel. Очистка пропусков обязана менять только
        то значение, которое является пропуском.
        """
        frame = pd.DataFrame(
            {"GGlevel": [3, 2, 1], "fulltime": ["undefined", "700", "800"]}
        )
        result = normalise_missing(frame)
        pd.testing.assert_series_equal(result["GGlevel"], frame["GGlevel"])


@pytest.fixture(scope="module")
def prepared():
    """Готовые признаки на реальной выгрузке — считаются один раз на модуль."""
    raw = pd.read_excel("Спектакль_utf8_2.xlsx", index_col=0)
    raw.columns = [c.replace("1TLDCDSILC_log_", "") for c in raw.columns]
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
    return build_features(raw)


class TestBuildFeatures:
    def test_no_rows_are_dropped(self, prepared):
        features, target, _ = prepared
        assert len(features) == 2796
        assert len(target) == 2796

    def test_target_distribution_is_intact(self, prepared):
        _, target, _ = prepared
        assert target.value_counts().sort_index().tolist() == [493, 845, 931, 527]

    def test_leaky_columns_are_absent(self, prepared):
        features, _, _ = prepared
        assert "GGfscore" not in features.columns
        assert "GGlevel" not in features.columns
        assert "Username" not in features.columns

    def test_column_types_partition_cleanly(self, prepared):
        features, _, _ = prepared
        numeric = set(numeric_columns(features))
        nominal = set(nominal_columns(features))
        assert numeric.isdisjoint(nominal)
        assert numeric | nominal == set(features.columns)

    def test_numeric_block_has_no_identifier_codes(self, prepared):
        """Коды объектов не должны попадать в числовой блок как величины."""
        features, _, _ = prepared
        for column in numeric_columns(features):
            assert not column.endswith("_first")

    def test_sequence_counts_are_consistent(self, prepared):
        features, _, _ = prepared
        actions = features["savepicture_n_actions"]
        repeats = features["savepicture_n_repeats"]
        assert (repeats <= actions).all()
        assert (actions >= 0).all()

    def test_missing_tokens_do_not_survive_as_categories(self, prepared):
        features, _, _ = prepared
        for column in nominal_columns(features):
            values = set(features[column].unique())
            assert not values & set(MISSING_TOKENS)

    def test_fulltime_is_in_minutes(self, prepared):
        features, _, _ = prepared
        finite = features["fulltime"].dropna()
        assert finite.max() < 120  # задание не длится дольше двух часов
        assert finite.min() >= 0
