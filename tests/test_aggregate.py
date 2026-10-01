import json
import math
from pathlib import Path

import pytest
from swissphenocam.cli.aggregate import (
    _classify_threshold,
    _nan_season_columns,
    _evaluate_filters,
    _build_report,
    _keep_row,
    _process_dataset,
)
from swissphenocam.cli.process import _drop_excluded_locations
from swissphenocam.config import AggregateConfig, load_aggregate_config


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


# --- allow_partial / exclude_locations ---------------------------------------

@pytest.mark.parametrize(
    "spring_ok,autumn_ok,allow_partial,expected",
    [
        (True, True, True, True),
        (False, True, True, True),
        (True, False, True, True),
        (False, False, True, False),
        (True, True, False, True),
        (False, True, False, False),
        (True, False, False, False),
        (False, False, False, False),
    ],
)
def test_keep_row(spring_ok, autumn_ok, allow_partial, expected):
    assert _keep_row(spring_ok, autumn_ok, allow_partial) is expected


def test_build_report_states_mode():
    assert "Per-season filters     : partial" in _build_report(10, 0, 0, 0, 0, [])
    assert "Per-season filters     : hard" in _build_report(
        10, 0, 0, 0, 0, [], allow_partial=False
    )


def test_drop_excluded_locations():
    units = [("aaa", "1", "2020", "i1"), ("bbb", "1", "2020", "i2"), ("aaa", "2", "2021", "i3")]
    kept, n = _drop_excluded_locations(units, ["aaa"])
    assert kept == [("bbb", "1", "2020", "i2")]
    assert n == 2


def _write_tree_year(root: Path, location: str, uid: str, missing_spring: int) -> None:
    full_name = f"{uid}_2020"
    d = root / f"{location}_1" / uid / "2020"
    d.mkdir(parents=True)
    meta = {"location": location, "sub_location": "1", "year": 2020, "full_name": full_name}
    (d / f"{full_name}_metadata.json").write_text(json.dumps(meta))
    flags = {"spring_max_consecutive_missing_days": missing_spring,
             "autumn_max_consecutive_missing_days": 0}
    (d / f"{full_name}-sampling-flags.json").write_text(json.dumps(flags))
    trans = {"dates": {"GU_50": 120.0, "GD_25": 280.0}, "snr": 20.0}
    (d / f"{full_name}-GCC-3D-p90-transitions.json").write_text(json.dumps(trans))


@pytest.fixture
def small_dataset(tmp_path):
    _write_tree_year(tmp_path, "keep", "i00001", missing_spring=3)   # passes
    _write_tree_year(tmp_path, "keep", "i00002", missing_spring=30)  # spring fails
    _write_tree_year(tmp_path, "drop", "i00003", missing_spring=3)   # excluded site
    return tmp_path


def _cfg(root: Path, **kw) -> AggregateConfig:
    return AggregateConfig(
        input_root=root, output_path=root / "out.csv",
        max_thresholds={"spring_max_consecutive_missing_days": 14}, **kw,
    )


def test_process_dataset_partial_mode_blanks_spring(small_dataset):
    df, _ = _process_dataset(_cfg(small_dataset), verbose=False)
    assert sorted(df["full_name"]) == ["i00001_2020", "i00002_2020", "i00003_2020"]
    row = df.set_index("full_name").loc["i00002_2020"]
    assert math.isnan(row["GCC_3D_p90-GU_50"])
    assert row["GCC_3D_p90-GD_25"] == 280.0


def test_process_dataset_hard_mode_excludes_tree_year(small_dataset):
    df, report = _process_dataset(_cfg(small_dataset, allow_partial=False), verbose=False)
    assert sorted(df["full_name"]) == ["i00001_2020", "i00003_2020"]
    assert "Partially accepted     :   0" in report


def test_process_dataset_exclude_locations(small_dataset):
    df, report = _process_dataset(_cfg(small_dataset, exclude_locations=["drop"]), verbose=False)
    assert "drop" not in set(df["location"])
    assert "Total tree-years found : 2" in report


def test_load_aggregate_config_reads_new_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"input_root": str(tmp_path), "output_path": str(tmp_path / "o.csv"),
                             "allow_partial": False, "exclude_locations": ["x"]}))
    cfg = load_aggregate_config(p)
    assert cfg.allow_partial is False
    assert cfg.exclude_locations == ["x"]


def test_process_dataset_required_keys(tmp_path):
    _write_tree_year(tmp_path, "keep", "i00001", missing_spring=3)
    _write_tree_year(tmp_path, "keep", "i00002", missing_spring=3)
    d = tmp_path / "keep_1" / "i00002" / "2020"
    trans = {"dates": {"GU_50": 120.0, "GD_25": None}, "uncertainties": {"GD_25_std": 2.0}}
    (d / "i00002_2020-GCC-3D-p90-transitions.json").write_text(json.dumps(trans))

    df, report = _process_dataset(_cfg(tmp_path), verbose=False)
    assert len(df) == 2  # not required: missing GD_25 is kept

    cfg = _cfg(tmp_path, required_keys=["GCC_3D_p90-GU_50", "GCC_3D_p90-GD_25"])
    df, report = _process_dataset(cfg, verbose=False)
    assert list(df["full_name"]) == ["i00001_2020"]
    assert "req  GCC_3D_p90-GD_25 present" in report
