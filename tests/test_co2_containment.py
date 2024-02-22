import numpy as np
import os
import pandas
from pathlib import Path
import pytest
import shapely.geometry

from ccs_scripts.co2_containment.co2_containment import (
    main,
)
from ccs_scripts.co2_containment.co2_calculation import (
    CalculationType,
    Co2Data,
    SourceData,
    _calculate_co2_data_from_source_data,
)


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


def _get_synthetic_case_paths():
    main_path = (
        Path(__file__).parents[1]
        / "tests"
        / "synthetic_model"
        / "realization-0"
        / "iter-0"
    )
    case_path = str(main_path / "eclipse" / "model" / "E_FLT_01-0")
    root_dir = "realization-0/iter-0"
    containment_polygon = str(
        main_path / "share" / "results" / "polygons" / "containment--boundary.csv"
    )
    hazardous_polygon = str(
        main_path / "share" / "results" / "polygons" / "hazardous--boundary.csv"
    )
    output_dir = str(main_path / "share" / "results" / "tables")
    return (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
    )


def test_synthetic_case_mass(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
    ) = _get_synthetic_case_paths()
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

    assert len(df) == 11
    assert df["date"].min() == "2025-01-01"
    assert df["date"].max() == "2500-01-01"
    assert df["total"].sum() == pytest.approx(17253002337.768658)
    assert df["total_gas"].sum() == pytest.approx(13426268976.322)
    assert df["total_aqueous"].sum() == pytest.approx(3826733361.4466577)
    assert df["total_contained"].sum() == pytest.approx(8488162800.155235)
    assert df["total_outside"].sum() == pytest.approx(7016268521.866909)
    assert df["total_hazardous"].sum() == pytest.approx(1748571015.7465131)
    assert df["gas_contained"].sum() == pytest.approx(6990346422.720001)
    assert df["aqueous_contained"].sum() == pytest.approx(1497816377.435234)
    assert df["gas_outside"].sum() == pytest.approx(5154504459.26111)
    assert df["aqueous_outside"].sum() == pytest.approx(1861764062.6057913)
    assert df["gas_hazardous"].sum() == pytest.approx(1281418094.3408813)
    assert df["aqueous_hazardous"].sum() == pytest.approx(467152921.40563196)


def test_synthetic_case_actual_volume(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
    ) = _get_synthetic_case_paths()

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

    assert len(df) == 11
    assert df["date"].min() == "2025-01-01"
    assert df["date"].max() == "2500-01-01"
    assert df["total"].sum() == pytest.approx(22071879.550792538)
    assert df["total_gas"].sum() == pytest.approx(19033465.136888713)
    assert df["total_aqueous"].sum() == pytest.approx(3038414.4139038236)
    assert df["total_contained"].sum() == pytest.approx(11046790.6125475)
    assert df["total_outside"].sum() == pytest.approx(8799634.69523845)
    assert df["total_hazardous"].sum() == pytest.approx(2225454.2430065842)
    assert df["gas_contained"].sum() == pytest.approx(9851596.627884675)
    assert df["aqueous_contained"].sum() == pytest.approx(1195193.9846628257)
    assert df["gas_outside"].sum() == pytest.approx(7328181.264653008)
    assert df["aqueous_outside"].sum() == pytest.approx(1471453.4305854416)
    assert df["gas_hazardous"].sum() == pytest.approx(1853687.2443510273)
    assert df["aqueous_hazardous"].sum() == pytest.approx(371766.9986555566)


def test_synthetic_case_cell_volume(mocker):
    (
        main_path,
        case_path,
        root_dir,
        containment_polygon,
        hazardous_polygon,
        output_dir,
    ) = _get_synthetic_case_paths()

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

    assert len(df) == 11
    assert df["date"].min() == "2025-01-01"
    assert df["date"].max() == "2500-01-01"
    assert df["total"].sum() == pytest.approx(2322814736)
    assert df["total_contained"].sum() == pytest.approx(270128250)
    assert df["total_outside"].sum() == pytest.approx(1817996448)
    assert df["total_hazardous"].sum() == pytest.approx(234690038)
