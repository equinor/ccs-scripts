import os
from pathlib import Path

import numpy as np
import pandas
import pytest

from ccs_scripts.co2_plume_extent.co2_plume_extent import (
    Configuration,
    LineDirection,
    _calculate_well_coordinates,
    _collect_results_into_dataframe,
    calculate_distances,
    main,
)


def test_calculate_well_coordinates():
    well_picks_path = str(
        Path(__file__).parents[1] / "tests" / "testdata_co2_plume" / "well_picks.csv"
    )
    x1, y1 = _calculate_well_coordinates("dummmy", "well1", well_picks_path)
    assert x1 == pytest.approx(4050.0)
    assert y1 == pytest.approx(4050.0)
    x2, y2 = _calculate_well_coordinates("dummmy", "well2", well_picks_path)
    assert x2 == pytest.approx(3000.0)
    assert y2 == pytest.approx(3000.0)


def test_calc_plume_extents():
    case_path = str(
        Path(__file__).parents[1]
        / "tests"
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0"
    )
    config = Configuration(
        config_file="",
        calculation_type="plume_extent",
        injection_point_info="[462500.0, 5933100.0]",
        name="",
        case=case_path,
    )
    sgas_results, _, _ = calculate_distances(
        case_path,
        config,
        threshold_sgas=0.1,
    )[0]
    assert len(sgas_results) == 4
    assert np.isnan(sgas_results[0][1])
    assert sgas_results[-1][1] == pytest.approx(1269.1237856341113)

    sgas_results_2, _, _ = calculate_distances(
        case_path,
        config,
    )[0]
    assert len(sgas_results_2) == 4
    assert np.isnan(sgas_results_2[-1][1])

    sgas_results_3, _, _ = calculate_distances(
        case_path,
        config,
        threshold_sgas=0.0001,
    )[0]
    assert len(sgas_results_3) == 4
    assert sgas_results_3[-1][1] == pytest.approx(2070.3444680185216)


def test_calc_distances_to_point():
    case_path = str(
        Path(__file__).parents[1]
        / "tests"
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0"
    )
    config = Configuration(
        config_file="",
        calculation_type="point",
        injection_point_info="[467000.0, 5934000.0]",
        name="",
        case=case_path,
    )
    sgas_results, _, _ = calculate_distances(
        case_path,
        config,
        threshold_sgas=0.1,
    )[0]
    assert len(sgas_results) == 4
    assert np.isnan(sgas_results[0][1])
    assert sgas_results[-1][1] == pytest.approx(4465.953446894702)


def test_calc_distances_to_line():
    case_path = str(
        Path(__file__).parents[1]
        / "tests"
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0"
    )
    config = Configuration(
        config_file="",
        calculation_type="line",
        injection_point_info="[east, 467000.0]",
        name="",
        case=case_path,
    )
    sgas_results, _, _ = calculate_distances(
        case_path,
        config,
        threshold_sgas=0.1,
    )[0]
    assert len(sgas_results) == 4
    assert np.isnan(sgas_results[0][1])
    assert sgas_results[-1][1] == pytest.approx(4331.578703680541)

    config.distance_calculations[0].direction = LineDirection.WEST
    sgas_results, _, _ = calculate_distances(
        case_path,
        config,
        threshold_sgas=0.1,
    )[0]
    assert len(sgas_results) == 4
    assert np.isnan(sgas_results[0][1])
    assert sgas_results[-1][1] == 0.0

    config.distance_calculations[0].direction = LineDirection.NORTH
    config.distance_calculations[0].x = None
    config.distance_calculations[0].y = 5934000.0
    sgas_results, _, _ = calculate_distances(
        case_path,
        config,
        threshold_sgas=0.1,
    )[0]
    assert len(sgas_results) == 4
    assert np.isnan(sgas_results[0][1])
    assert sgas_results[-1][1] == pytest.approx(498.06528236251324)

    config.distance_calculations[0].direction = LineDirection.SOUTH
    sgas_results, _, _ = calculate_distances(
        case_path,
        config,
        threshold_sgas=0.1,
    )[0]
    assert len(sgas_results) == 4
    assert np.isnan(sgas_results[0][1])
    assert sgas_results[-1][1] == pytest.approx(0.0)


def test_export_to_csv():
    case_path = str(
        Path(__file__).parents[1]
        / "tests"
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0"
    )
    config = Configuration(
        config_file="",
        calculation_type="plume_extent",
        injection_point_info="[462500.0, 5933100.0]",
        name="",
        case=case_path,
    )
    all_results = calculate_distances(
        case_path,
        config,
        threshold_sgas=0.1,
    )

    out_file = "temp.csv"
    df = _collect_results_into_dataframe(all_results, config)
    df.to_csv(out_file, index=False)

    df = pandas.read_csv(out_file)
    assert "MAX_DISTANCE_plume_extent_1_SGAS" in df.keys()
    assert "MAX_DISTANCE_plume_extent_1_AMFG" not in df.keys()
    assert df["MAX_DISTANCE_plume_extent_1_SGAS"].iloc[-1] == pytest.approx(
        1269.1237856341113
    )

    os.remove(out_file)


def test_plume_extent(mocker):
    case_path = str(
        Path(__file__).parents[1]
        / "tests"
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0"
    )
    output_path = str(
        Path(__file__).parents[1] / "tests" / "testdata_co2_plume" / "plume_extent.csv"
    )
    mocker.patch(
        "sys.argv",
        [
            "--case",
            case_path,
            "--injection_point_info",
            "[462500.0,5933100.0]",
            "--threshold_sgas",
            "0.02",
            "--output",
            output_path,
        ],
    )
    main()

    df = pandas.read_csv(output_path)
    assert "MAX_DISTANCE_plume_extent_1_SGAS" in df.keys()
    assert "MAX_DISTANCE_plume_extent_1_AMFG" not in df.keys()
    assert df["MAX_DISTANCE_plume_extent_1_SGAS"].iloc[-1] == pytest.approx(
        1915.5936794783647
    )

    os.remove(output_path)


def _get_synthetic_case_paths(case: str):
    file_name = ""
    if case == "eclipse":
        file_name = "E_FLT_01-0"
    elif case == "pflotran":
        file_name = "P_FLT_01-0"
    case_path = str(
        Path(__file__).parents[1]
        / "tests"
        / "synthetic_model"
        / "realization-0"
        / "iter-0"
        / case
        / "model"
        / file_name
    )
    output_path = str(
        Path(__file__).parents[1] / "tests" / "testdata_co2_plume" / "plume_extent.csv"
    )
    return (case_path, output_path)


def test_plume_extent_eclipse_using_well_name(mocker):
    (case_path, output_path) = _get_synthetic_case_paths("eclipse")
    mocker.patch(
        "sys.argv",
        [
            "--case",
            case_path,
            "--injection_point_info",
            "INJ",
            "--threshold_sgas",
            "0.015",
            "--threshold_amfg",
            "0.0004",
            "--output",
            output_path,
        ],
    )
    main()

    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "plume_extent"
        / "plume_extent_eclipse.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = df.sort_values("date")
    df_answer = df_answer.sort_values("date")
    pandas.testing.assert_frame_equal(df, df_answer)


def test_plume_extent_eclipse_using_coordinates(mocker):
    (case_path, output_path) = _get_synthetic_case_paths("eclipse")
    mocker.patch(
        "sys.argv",
        [
            "--case",
            case_path,
            "--injection_point_info",
            "[2124.95, 2108.24]",
            "--threshold_sgas",
            "0.015",
            "--threshold_amfg",
            "0.0004",
            "--output",
            output_path,
        ],
    )
    main()

    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "plume_extent"
        / "plume_extent_eclipse.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = df.sort_values("date")
    df_answer = df_answer.sort_values("date")
    pandas.testing.assert_frame_equal(df, df_answer)


def test_plume_extent_eclipse_using_coordinates_small_thresholds(mocker):
    (case_path, output_path) = _get_synthetic_case_paths("eclipse")
    mocker.patch(
        "sys.argv",
        [
            "--case",
            case_path,
            "--injection_point_info",
            "[2124.95, 2108.24]",
            "--threshold_sgas",
            "0.000000001",
            "--threshold_amfg",
            "0.000000001",
            "--output",
            output_path,
        ],
    )
    main()

    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "plume_extent"
        / "plume_extent_eclipse_small_thresholds.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = df.sort_values("date")
    df_answer = df_answer.sort_values("date")
    pandas.testing.assert_frame_equal(df, df_answer)


def test_plume_extent_pflotran_using_well_name(mocker):
    (case_path, output_path) = _get_synthetic_case_paths("pflotran")
    mocker.patch(
        "sys.argv",
        [
            "--case",
            case_path,
            "--injection_point_info",
            "INJ",
            "--threshold_sgas",
            "0.015",
            "--threshold_amfg",
            "0.0004",
            "--output",
            output_path,
        ],
    )
    main()

    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "plume_extent"
        / "plume_extent_pflotran.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = df.sort_values("date")
    df_answer = df_answer.sort_values("date")
    pandas.testing.assert_frame_equal(df, df_answer)


def test_plume_extent_pflotran_using_coordinates(mocker):
    (case_path, output_path) = _get_synthetic_case_paths("pflotran")
    mocker.patch(
        "sys.argv",
        [
            "--case",
            case_path,
            "--injection_point_info",
            "[2124.95, 2108.24]",
            "--threshold_sgas",
            "0.015",
            "--threshold_amfg",
            "0.0004",
            "--output",
            output_path,
        ],
    )
    main()

    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "plume_extent"
        / "plume_extent_pflotran.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = df.sort_values("date")
    df_answer = df_answer.sort_values("date")
    pandas.testing.assert_frame_equal(df, df_answer)


def test_plume_extent_pflotran_using_coordinates_default_thresholds(mocker):
    (case_path, output_path) = _get_synthetic_case_paths("pflotran")
    mocker.patch(
        "sys.argv",
        [
            "--case",
            case_path,
            "--injection_point_info",
            "[2124.95, 2108.24]",
            "--output",
            output_path,
        ],
    )
    main()

    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "plume_extent"
        / "plume_extent_pflotran_default_thresholds.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = df.sort_values("date")
    df_answer = df_answer.sort_values("date")
    pandas.testing.assert_frame_equal(df, df_answer)


def test_yaml_file_pflotran(mocker):
    (case_path, output_path) = _get_synthetic_case_paths("pflotran")
    config_path = str(
        Path(__file__).parents[1] / "tests" / "yaml" / "config_co2_plume_extent.yml"
    )
    mocker.patch(
        "sys.argv",
        [
            "--case",
            case_path,
            "--config_file",
            config_path,
            "--output",
            output_path,
        ],
    )
    main()

    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "plume_extent"
        / "plume_extent_pflotran_yaml_file.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = df.sort_values("date")
    df_answer = df_answer.sort_values("date")
    pandas.testing.assert_frame_equal(df, df_answer)
