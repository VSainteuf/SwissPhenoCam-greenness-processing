import math
import pytest
from swissphenocam.cli.aggregate import (
    _classify_threshold,
    _nan_season_columns,
    _evaluate_filters,
    _build_report,
)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("spring_coverage_ratio", "spring"),
        ("spring_max_consecutive_missing_days", "spring"),
        ("autumn_coverage_ratio", "autumn"),
        ("autumn_max_gap_days", "autumn"),
        ("GCC_3D_p90-GU_50-std", "spring"),
        ("GCC_3D_p90-GD_25-std", "autumn"),
        ("GCC_3D_p90-snr", "whole_year"),
        ("n_peak", "whole_year"),
        ("summer_coverage_ratio", "whole_year"),
    ],
)
def test_classify_threshold(key, expected):
    assert _classify_threshold(key) == expected


def test_nan_season_columns_gu():
    row = {
        "full_name": "Tree0001",
        "GCC_3D_p90-PEAK": 180,
        "GCC_3D_p90-GU_10": 100,
        "GCC_3D_p90-GU_50": 120,
        "GCC_3D_p90-GU_10-std": 2.0,
        "GCC_3D_p90-GD_10": 250,
        "GCC_3D_p90-GD_50": 270,
        "GCC_3D_p90-GD_10-std": 3.0,
    }
    _nan_season_columns(row, "GU")
    assert math.isnan(row["GCC_3D_p90-GU_10"])
    assert math.isnan(row["GCC_3D_p90-GU_50"])
    assert math.isnan(row["GCC_3D_p90-GU_10-std"])
    assert row["GCC_3D_p90-PEAK"] == 180
    assert row["GCC_3D_p90-GD_10"] == 250
    assert row["GCC_3D_p90-GD_50"] == 270
    assert row["GCC_3D_p90-GD_10-std"] == 3.0
    assert row["full_name"] == "Tree0001"


def test_nan_season_columns_gd():
    row = {
        "full_name": "Tree0001",
        "GCC_3D_p90-PEAK": 180,
        "GCC_3D_p90-GU_10": 100,
        "GCC_3D_p90-GU_50": 120,
        "GCC_3D_p90-GU_10-std": 2.0,
        "GCC_3D_p90-GD_10": 250,
        "GCC_3D_p90-GD_50": 270,
        "GCC_3D_p90-GD_10-std": 3.0,
    }
    _nan_season_columns(row, "GD")
    assert math.isnan(row["GCC_3D_p90-GD_10"])
    assert math.isnan(row["GCC_3D_p90-GD_50"])
    assert math.isnan(row["GCC_3D_p90-GD_10-std"])
    assert row["GCC_3D_p90-PEAK"] == 180
    assert row["GCC_3D_p90-GU_10"] == 100
    assert row["GCC_3D_p90-GU_50"] == 120
    assert row["GCC_3D_p90-GU_10-std"] == 2.0
    assert row["full_name"] == "Tree0001"


def test_evaluate_filters_all_pass():
    row = {
        "spring_coverage_ratio": 0.9,
        "autumn_coverage_ratio": 0.9,
        "GCC_3D_p90-snr": 20.0,
    }
    min_thresholds = {
        "spring_coverage_ratio": 0.75,
        "autumn_coverage_ratio": 0.75,
        "GCC_3D_p90-snr": 12.0,
    }
    max_thresholds = {}
    assert _evaluate_filters(row, min_thresholds, max_thresholds) == (True, True, True)


def test_evaluate_filters_spring_fail():
    row = {
        "spring_coverage_ratio": 0.5,
        "autumn_coverage_ratio": 0.9,
        "GCC_3D_p90-snr": 20.0,
    }
    min_thresholds = {
        "spring_coverage_ratio": 0.75,
        "autumn_coverage_ratio": 0.75,
        "GCC_3D_p90-snr": 12.0,
    }
    max_thresholds = {}
    assert _evaluate_filters(row, min_thresholds, max_thresholds) == (True, False, True)


def test_evaluate_filters_autumn_fail():
    row = {
        "spring_coverage_ratio": 0.9,
        "autumn_coverage_ratio": 0.5,
        "GCC_3D_p90-snr": 20.0,
    }
    min_thresholds = {
        "spring_coverage_ratio": 0.75,
        "autumn_coverage_ratio": 0.75,
        "GCC_3D_p90-snr": 12.0,
    }
    max_thresholds = {}
    assert _evaluate_filters(row, min_thresholds, max_thresholds) == (True, True, False)


def test_evaluate_filters_both_fail():
    row = {
        "spring_coverage_ratio": 0.5,
        "autumn_coverage_ratio": 0.5,
        "GCC_3D_p90-snr": 20.0,
    }
    min_thresholds = {
        "spring_coverage_ratio": 0.75,
        "autumn_coverage_ratio": 0.75,
        "GCC_3D_p90-snr": 12.0,
    }
    max_thresholds = {}
    assert _evaluate_filters(row, min_thresholds, max_thresholds) == (True, False, False)


def test_evaluate_filters_whole_year_fail():
    row = {
        "spring_coverage_ratio": 0.9,
        "autumn_coverage_ratio": 0.9,
        "GCC_3D_p90-snr": 5.0,
    }
    min_thresholds = {
        "spring_coverage_ratio": 0.75,
        "autumn_coverage_ratio": 0.75,
        "GCC_3D_p90-snr": 12.0,
    }
    max_thresholds = {}
    assert _evaluate_filters(row, min_thresholds, max_thresholds) == (False, True, True)


def test_evaluate_filters_no_season_filters():
    row = {
        "GCC_3D_p90-snr": 15.0,
        "GCC_3D_p90-n_peak": 1,
    }
    min_thresholds = {"GCC_3D_p90-snr": 12.0}
    max_thresholds = {"n_peak": 1}
    assert _evaluate_filters(row, min_thresholds, max_thresholds) == (True, True, True)


def test_build_report_with_partials():
    report = _build_report(
        100,
        10,
        0,
        5,
        3,
        [("min  snr >= 12.0", 10)],
    )
    assert "Fully accepted" in report
    assert "Partially accepted" in report
    assert "Spring removed" in report
    assert "Autumn removed" in report
    assert "Excluded (thresholds)" in report
    assert "Excluded (external CSV)" in report
