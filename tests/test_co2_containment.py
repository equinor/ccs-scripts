import os
from pathlib import Path

import numpy as np
import pandas
import pytest
import shapely.geometry

from ccs_scripts.co2_containment.co2_calculation import (
    CalculationType,
    Co2Data,
    SourceData,
    _calculate_co2_data_from_source_data,
)
from ccs_scripts.co2_containment.co2_containment import main

REGION_PROPERTY = "FIPREG"


def _simple_cube_grid():
    """
    Create simple cube grid
    """
    dims = (13, 17, 19)
    m_x, m_y, m_z = np.meshgrid(
        np.linspace(-1, 1, dims[0]),
        np.linspace(-1, 1, dims[1]),
        np.linspace(-1, 1, dims[2]),
        indexing="ij",
    )
    dates = [f"{d}0101" for d in range(2030, 2050)]
    dists = np.sqrt(m_x**2 + m_y**2 + m_z**2)
    gas_saturations = {}
    for count, date in enumerate(dates):
        gas_saturations[date] = np.maximum(
            np.exp(-3 * (dists.flatten() / ((count + 1) / len(dates))) ** 2) - 0.05, 0.0
        )
    size = np.prod(dims)
    return SourceData(
        m_x.flatten(),
        m_y.flatten(),
        PORV={date: np.ones(size) * 0.3 for date in dates},
        VOL={date: np.ones(size) * (8 / size) for date in dates},
        DATES=dates,
        DWAT={date: np.ones(size) * 1000.0 for date in dates},
        SWAT={date: 1 - value for date, value in gas_saturations.items()},
        SGAS=gas_saturations,
        DGAS={date: np.ones(size) * 100.0 for date in dates},
        AMFG={
            date: np.ones(size) * 0.02 * value
            for date, value in gas_saturations.items()
        },
        YMFG={date: np.ones(size) * 0.99 for date in dates},
    )


def _simple_cube_grid_eclipse():
    """
    Create simple cube grid, eclipse properties
    """
    dims = (13, 17, 19)
    m_x, m_y, m_z = np.meshgrid(
        np.linspace(-1, 1, dims[0]),
        np.linspace(-1, 1, dims[1]),
        np.linspace(-1, 1, dims[2]),
        indexing="ij",
    )
    dates = [f"{d}0101" for d in range(2030, 2050)]
    dists = np.sqrt(m_x**2 + m_y**2 + m_z**2)
    gas_saturations = {}
    for count, date in enumerate(dates):
        gas_saturations[date] = np.maximum(
            np.exp(-3 * (dists.flatten() / ((count + 1) / len(dates))) ** 2) - 0.05, 0.0
        )
    size = np.prod(dims)
    return SourceData(
        m_x.flatten(),
        m_y.flatten(),
        RPORV={date: np.ones(size) * 0.3 for date in dates},
        VOL={date: np.ones(size) * (8 / size) for date in dates},
        DATES=dates,
        BWAT={date: np.ones(size) * 1000.0 for date in dates},
        SWAT={date: 1 - value for date, value in gas_saturations.items()},
        SGAS=gas_saturations,
        BGAS={date: np.ones(size) * 100.0 for date in dates},
        XMF2={
            date: np.ones(size) * 0.02 * value
            for date, value in gas_saturations.items()
        },
        YMF2={date: np.ones(size) * 0.99 for date in dates},
    )


def _simple_poly():
    """
    Create simple polygon
    """
    return shapely.geometry.Polygon(
        np.array(
            [
                [-0.45, -0.38],
                [0.41, -0.39],
                [0.33, 0.76],
                [-0.27, 0.75],
                [-0.45, -0.38],
            ]
        )
    )


def test_simple_cube_grid():
    """
    Test simple cube grid. Testing result for last date.
    """
    simple_cube_grid = _simple_cube_grid()

    co2_data = _calculate_co2_data_from_source_data(
        simple_cube_grid,
        CalculationType.MASS,
    )
    assert len(co2_data.data_list) == len(simple_cube_grid.DATES)
    assert co2_data.units == "kg"
    assert co2_data.data_list[-1].date == "20490101"
    assert co2_data.data_list[-1].gas_phase.sum() == pytest.approx(9585.032869548137)
    assert co2_data.data_list[-1].aqu_phase.sum() == pytest.approx(2834.956447728449)

    simple_cube_grid_eclipse = _simple_cube_grid_eclipse()

    co2_data_eclipse = _calculate_co2_data_from_source_data(
        simple_cube_grid_eclipse,
        CalculationType.MASS,
    )
    assert len(co2_data_eclipse.data_list) == len(simple_cube_grid_eclipse.DATES)
    assert co2_data_eclipse.units == "kg"
    assert co2_data_eclipse.data_list[-1].date == "20490101"
    assert co2_data_eclipse.data_list[-1].gas_phase.sum() == pytest.approx(
        419249.33771403536
    )
    assert co2_data_eclipse.data_list[-1].aqu_phase.sum() == pytest.approx(
        51468.54223011175
    )


def test_zoned_simple_cube_grid():
    """
    Create simple cube grid, zoned. Testing result for last date.
    """
    simple_cube_grid = _simple_cube_grid()

    # pylint: disable-next=no-member
    random_state = np.random.RandomState(123)
    zone = random_state.choice([1, 2, 3], size=simple_cube_grid.PORV["20300101"].shape)
    simple_cube_grid.zone = zone
    co2_data = _calculate_co2_data_from_source_data(
        simple_cube_grid,
        CalculationType.MASS,
    )
    assert isinstance(co2_data, Co2Data)
    assert co2_data.data_list[-1].date == "20490101"
    assert co2_data.data_list[-1].gas_phase.sum() == pytest.approx(9585.032869548137)
    assert co2_data.data_list[-1].aqu_phase.sum() == pytest.approx(2834.956447728449)


def _get_synthetic_case_paths(case: str):
    file_name = ""
    if case == "eclipse":
        file_name = "E_FLT_01-0"
    elif case == "pflotran":
        file_name = "P_FLT_01-0"
    main_path = (
        Path(__file__).parents[1]
        / "tests"
        / "synthetic_model"
        / "realization-0"
        / "iter-0"
    )
    case_path = str(main_path / case / "model" / file_name)
    root_dir = "realization-0/iter-0"
    containment_polygon = str(
        main_path / "share" / "results" / "polygons" / "containment--boundary.csv"
    )
    hazardous_polygon = str(
        main_path / "share" / "results" / "polygons" / "hazardous--boundary.csv"
    )
    output_dir = str(main_path / "share" / "results" / "tables")
    zone_file_path = str(main_path / "rms" / "zone" / "zonation_ecl_map.yml")
    return (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
        zone_file_path,
    )


def _sort_dataframe(df: pandas.DataFrame):
    if "zone" in df and "region" in df:
        df = df.sort_values(["date", "zone", "region"])
    elif "zone" in df:
        df = df.sort_values(["date", "zone"])
    elif "region" in df:
        df = df.sort_values(["date", "region"])
    else:
        df = df.sort_values("date")
    return df


def test_synthetic_case_eclipse_mass(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
        zone_file_path,
    ) = _get_synthetic_case_paths("eclipse")
    args = [
        "sys.argv",
        case_path,
        "mass",
        "--root_dir",
        root_dir,
        "--out_dir",
        output_dir,
        "--containment_polygon",
        containment_polygon,
        "--hazardous_polygon",
        hazardous_polygon,
        "--zonefile",
        zone_file_path,
        "--region_property",
        REGION_PROPERTY,
    ]
    mocker.patch(
        "sys.argv",
        args,
    )
    main()

    output_path = str(main_path / "share" / "results" / "tables" / "plume_mass.csv")
    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0] / "answers" / "containment" / "plume_mass_eclipse.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = _sort_dataframe(df)
    df_answer = _sort_dataframe(df_answer)
    pandas.testing.assert_frame_equal(df, df_answer)


def test_synthetic_case_eclipse_actual_volume(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
        zone_file_path,
    ) = _get_synthetic_case_paths("eclipse")

    args = [
        "sys.argv",
        case_path,
        "actual_volume",
        "--root_dir",
        root_dir,
        "--out_dir",
        output_dir,
        "--containment_polygon",
        containment_polygon,
        "--hazardous_polygon",
        hazardous_polygon,
        "--zonefile",
        zone_file_path,
        "--region_property",
        REGION_PROPERTY,
    ]
    mocker.patch(
        "sys.argv",
        args,
    )
    main()

    output_path = str(
        main_path / "share" / "results" / "tables" / "plume_actual_volume.csv"
    )
    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "containment"
        / "plume_actual_volume_eclipse.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = _sort_dataframe(df)
    df_answer = _sort_dataframe(df_answer)
    pandas.testing.assert_frame_equal(df, df_answer)


def test_synthetic_case_eclipse_cell_volume(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
        zone_file_path,
    ) = _get_synthetic_case_paths("eclipse")

    args = [
        "sys.argv",
        case_path,
        "cell_volume",
        "--root_dir",
        root_dir,
        "--out_dir",
        output_dir,
        "--containment_polygon",
        containment_polygon,
        "--hazardous_polygon",
        hazardous_polygon,
        "--zonefile",
        zone_file_path,
        "--region_property",
        REGION_PROPERTY,
    ]
    mocker.patch(
        "sys.argv",
        args,
    )
    main()

    output_path = str(
        main_path / "share" / "results" / "tables" / "plume_cell_volume.csv"
    )
    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "containment"
        / "plume_cell_volume_eclipse.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = _sort_dataframe(df)
    df_answer = _sort_dataframe(df_answer)
    pandas.testing.assert_frame_equal(df, df_answer)


def test_synthetic_case_eclipse_mass_no_zones(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
        dummy,
    ) = _get_synthetic_case_paths("eclipse")
    args = [
        "sys.argv",
        case_path,
        "mass",
        "--root_dir",
        root_dir,
        "--out_dir",
        output_dir,
        "--containment_polygon",
        containment_polygon,
        "--hazardous_polygon",
        hazardous_polygon,
        "--region_property",
        REGION_PROPERTY,
    ]
    mocker.patch(
        "sys.argv",
        args,
    )
    main()

    output_path = str(main_path / "share" / "results" / "tables" / "plume_mass.csv")
    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "containment"
        / "plume_mass_eclipse_no_zones.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = _sort_dataframe(df)
    df_answer = _sort_dataframe(df_answer)
    pandas.testing.assert_frame_equal(df, df_answer)


def test_synthetic_case_eclipse_mass_no_regions(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
        zone_file_path,
    ) = _get_synthetic_case_paths("eclipse")
    args = [
        "sys.argv",
        case_path,
        "mass",
        "--root_dir",
        root_dir,
        "--out_dir",
        output_dir,
        "--containment_polygon",
        containment_polygon,
        "--hazardous_polygon",
        hazardous_polygon,
        "--zonefile",
        zone_file_path,
    ]
    mocker.patch(
        "sys.argv",
        args,
    )
    main()

    output_path = str(main_path / "share" / "results" / "tables" / "plume_mass.csv")
    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "containment"
        / "plume_mass_eclipse_no_regions.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = _sort_dataframe(df)
    df_answer = _sort_dataframe(df_answer)
    pandas.testing.assert_frame_equal(df, df_answer)


def test_synthetic_case_eclipse_mass_no_zones_no_regions(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
        dummy,
    ) = _get_synthetic_case_paths("eclipse")
    args = [
        "sys.argv",
        case_path,
        "mass",
        "--root_dir",
        root_dir,
        "--out_dir",
        output_dir,
        "--containment_polygon",
        containment_polygon,
        "--hazardous_polygon",
        hazardous_polygon,
    ]
    mocker.patch(
        "sys.argv",
        args,
    )
    main()

    output_path = str(main_path / "share" / "results" / "tables" / "plume_mass.csv")
    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "containment"
        / "plume_mass_eclipse_no_zones_no_regions.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = _sort_dataframe(df)
    df_answer = _sort_dataframe(df_answer)
    pandas.testing.assert_frame_equal(df, df_answer)


def test_synthetic_case_pflotran_mass(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
        zone_file_path,
    ) = _get_synthetic_case_paths("pflotran")
    args = [
        "sys.argv",
        case_path,
        "mass",
        "--root_dir",
        root_dir,
        "--out_dir",
        output_dir,
        "--containment_polygon",
        containment_polygon,
        "--hazardous_polygon",
        hazardous_polygon,
        "--zonefile",
        zone_file_path,
        "--region_property",
        REGION_PROPERTY,
    ]
    mocker.patch(
        "sys.argv",
        args,
    )
    main()

    output_path = str(main_path / "share" / "results" / "tables" / "plume_mass.csv")
    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "containment"
        / "plume_mass_pflotran.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = _sort_dataframe(df)
    df_answer = _sort_dataframe(df_answer)
    pandas.testing.assert_frame_equal(df, df_answer)


def test_synthetic_case_pflotran_actual_volume(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
        zone_file_path,
    ) = _get_synthetic_case_paths("pflotran")
    args = [
        "sys.argv",
        case_path,
        "actual_volume",
        "--root_dir",
        root_dir,
        "--out_dir",
        output_dir,
        "--containment_polygon",
        containment_polygon,
        "--hazardous_polygon",
        hazardous_polygon,
        "--zonefile",
        zone_file_path,
        "--region_property",
        REGION_PROPERTY,
    ]
    mocker.patch(
        "sys.argv",
        args,
    )
    main()

    output_path = str(
        main_path / "share" / "results" / "tables" / "plume_actual_volume.csv"
    )
    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "containment"
        / "plume_actual_volume_pflotran.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = _sort_dataframe(df)
    df_answer = _sort_dataframe(df_answer)
    pandas.testing.assert_frame_equal(df, df_answer)


def test_synthetic_case_pflotran_cell_volume(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
        zone_file_path,
    ) = _get_synthetic_case_paths("pflotran")
    args = [
        "sys.argv",
        case_path,
        "cell_volume",
        "--root_dir",
        root_dir,
        "--out_dir",
        output_dir,
        "--containment_polygon",
        containment_polygon,
        "--hazardous_polygon",
        hazardous_polygon,
        "--zonefile",
        zone_file_path,
        "--region_property",
        REGION_PROPERTY,
    ]
    mocker.patch(
        "sys.argv",
        args,
    )
    main()

    output_path = str(
        main_path / "share" / "results" / "tables" / "plume_cell_volume.csv"
    )
    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "containment"
        / "plume_cell_volume_pflotran.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = _sort_dataframe(df)
    df_answer = _sort_dataframe(df_answer)
    pandas.testing.assert_frame_equal(df, df_answer)


def test_synthetic_case_pflotran_mass_residual_trapping(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
        zone_file_path,
    ) = _get_synthetic_case_paths("pflotran")
    args = [
        "sys.argv",
        case_path,
        "mass",
        "--root_dir",
        root_dir,
        "--out_dir",
        output_dir,
        "--containment_polygon",
        containment_polygon,
        "--hazardous_polygon",
        hazardous_polygon,
        "--zonefile",
        zone_file_path,
        "--region_property",
        REGION_PROPERTY,
        "--residual_trapping",
    ]
    mocker.patch(
        "sys.argv",
        args,
    )
    main()

    output_path = str(main_path / "share" / "results" / "tables" / "plume_mass.csv")
    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "containment"
        / "plume_mass_pflotran_residual_trapping.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = _sort_dataframe(df)
    df_answer = _sort_dataframe(df_answer)
    pandas.testing.assert_frame_equal(df, df_answer)


def test_synthetic_case_pflotran_actual_volume_residual_trapping(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
        zone_file_path,
    ) = _get_synthetic_case_paths("pflotran")
    args = [
        "sys.argv",
        case_path,
        "actual_volume",
        "--root_dir",
        root_dir,
        "--out_dir",
        output_dir,
        "--containment_polygon",
        containment_polygon,
        "--hazardous_polygon",
        hazardous_polygon,
        "--zonefile",
        zone_file_path,
        "--region_property",
        REGION_PROPERTY,
        "--residual_trapping",
    ]
    mocker.patch(
        "sys.argv",
        args,
    )
    main()

    output_path = str(
        main_path / "share" / "results" / "tables" / "plume_actual_volume.csv"
    )
    df = pandas.read_csv(output_path)
    os.remove(output_path)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "containment"
        / "plume_actual_volume_pflotran_residual_trapping.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df = _sort_dataframe(df)
    df_answer = _sort_dataframe(df_answer)
    pandas.testing.assert_frame_equal(df, df_answer)
