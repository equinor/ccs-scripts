import os
from pathlib import Path

import pandas
import pytest

from ccs_scripts.co2_plume_area.co2_plume_area import calculate_plume_area, main


def test_calc_plume_area():
    input_path = str(
        Path(__file__).parents[1] / "tests" / "testdata_co2_plume" / "surfaces"
    )
    out = calculate_plume_area(input_path, "SGAS")
    assert len(out) == 3
    results = [x[1] for x in out]
    results.sort()
    assert results[0] == 0.0
    assert results[1] == pytest.approx(120000.0)
    assert results[2] == pytest.approx(285000.0)


def test_plume_area(mocker):
    input_path = str(
        Path(__file__).parents[1] / "tests" / "testdata_co2_plume" / "surfaces"
    )
    output_path = str(
        Path(__file__).parents[1] / "tests" / "testdata_co2_plume" / "plume_area.csv"
    )
    mocker.patch("sys.argv", ["--input", input_path, "--output", output_path])
    main()

    df = pandas.read_csv(output_path)
    os.remove(output_path)

    assert "formation_SGAS" in df.keys()
    assert "formation_AMFG" not in df.keys()
    assert df["formation_SGAS"].iloc[-1] == pytest.approx(285000.0)


def _get_synthetic_case_paths(case: str):
    dir_name = "surfaces_synthetic_case_" + case
    input_path = str(
        Path(__file__).parents[1] / "tests" / "testdata_co2_plume" / dir_name
    )
    output_path = str(
        Path(__file__).parents[1] / "tests" / "testdata_co2_plume" / "plume_area.csv"
    )
    return (input_path, output_path)


def test_plume_area_synthetic_case_eclipse(mocker):
    (input_path, output_path) = _get_synthetic_case_paths("eclipse")
    mocker.patch("sys.argv", ["--input", input_path, "--output", output_path])
    main()

    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0] / "answers" / "plume_area" / "plume_area_eclipse.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = df.sort_values("date")
    df_answer = df_answer.sort_values("date")
    pandas.testing.assert_frame_equal(df, df_answer)


def test_plume_area_synthetic_case_pflotran(mocker):
    (input_path, output_path) = _get_synthetic_case_paths("pflotran")
    mocker.patch("sys.argv", ["--input", input_path, "--output", output_path])
    main()

    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0] / "answers" / "plume_area" / "plume_area_pflotran.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = df.sort_values("date")
    df_answer = df_answer.sort_values("date")
    pandas.testing.assert_frame_equal(df, df_answer)
