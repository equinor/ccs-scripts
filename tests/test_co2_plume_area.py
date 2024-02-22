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


def _get_expected_columns(case: str) -> list[str]:
    columns = ["date", "all_SGAS", "amethyst_SGAS", "ruby_SGAS", "topaz_SGAS"]
    if case == "eclipse":
        columns += ["all_XMF2", "amethyst_XMF2", "ruby_XMF2", "topaz_XMF2"]
    elif case == "pflotran":
        columns += ["all_AMFG", "amethyst_AMFG", "ruby_AMFG", "topaz_AMFG"]
    return columns


def test_plume_area_synthetic_case_eclipse(mocker):
    (input_path, output_path) = _get_synthetic_case_paths("eclipse")
    mocker.patch("sys.argv", ["--input", input_path, "--output", output_path])
    main()

    df = pandas.read_csv(output_path)
    os.remove(output_path)

    assert len(df) == 11
    for c in _get_expected_columns("eclipse"):
        assert c in df
    assert "all_AMFG" not in df.keys()

    assert df["all_SGAS"].iloc[-1] == pytest.approx(2710000.0)
    assert df["amethyst_SGAS"].iloc[-1] == pytest.approx(2620000.0)
    assert df["ruby_SGAS"].iloc[-1] == pytest.approx(1220000.0)
    assert df["topaz_SGAS"].iloc[-1] == pytest.approx(1150000.0)
    assert df["all_XMF2"].iloc[-1] == pytest.approx(7350000.0)
    assert df["amethyst_XMF2"].iloc[-1] == pytest.approx(6970000.0)
    assert df["ruby_XMF2"].iloc[-1] == pytest.approx(6720000.0)
    assert df["topaz_XMF2"].iloc[-1] == pytest.approx(7210000.0)


def test_plume_area_synthetic_case_pflotran(mocker):
    (input_path, output_path) = _get_synthetic_case_paths("pflotran")
    mocker.patch("sys.argv", ["--input", input_path, "--output", output_path])
    main()

    df = pandas.read_csv(output_path)
    os.remove(output_path)

    assert len(df) == 31
    for c in _get_expected_columns("pflotran"):
        assert c in df
    assert "all_XMF2" not in df.keys()
    assert df["all_SGAS"].iloc[-1] == pytest.approx(2790000.0)
    assert df["amethyst_SGAS"].iloc[-1] == pytest.approx(2700000.0)
    assert df["ruby_SGAS"].iloc[-1] == pytest.approx(810000.0)
    assert df["topaz_SGAS"].iloc[-1] == pytest.approx(740000.0)
    assert df["all_AMFG"].iloc[-1] == pytest.approx(7890000.0)
    assert df["amethyst_AMFG"].iloc[-1] == pytest.approx(7800000.0)
    assert df["ruby_AMFG"].iloc[-1] == pytest.approx(7280000.0)
    assert df["topaz_AMFG"].iloc[-1] == pytest.approx(7760000.0)
