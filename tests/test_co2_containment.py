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


ZONES_SYNTHETIC_CASE = ["all", "Amethyst", "Ruby", "Topaz"]


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
    if "zone" and "region" in df:
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
        / "plume_mass_eclipse.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    pandas.testing.assert_frame_equal(df, df_answer)

    assert len(df) == 66
    assert len(df["date"].unique()) == 11
    assert df["date"].min() == "2025-01-01"
    assert df["date"].max() == "2500-01-01"

    df_all = df[(df["zone"] == "all") & (df["region"] == "all")]
    assert df_all["total"].sum() == pytest.approx(17253002337.768658)
    assert df_all["total_gas"].sum() == pytest.approx(13426268976.322)
    assert df_all["total_aqueous"].sum() == pytest.approx(3826733361.4466577)
    assert df_all["total_contained"].sum() == pytest.approx(8488162800.155235)
    assert df_all["total_outside"].sum() == pytest.approx(7016268521.866909)
    assert df_all["total_hazardous"].sum() == pytest.approx(1748571015.7465131)
    assert df_all["gas_contained"].sum() == pytest.approx(6990346422.720001)
    assert df_all["aqueous_contained"].sum() == pytest.approx(1497816377.435234)
    assert df_all["gas_outside"].sum() == pytest.approx(5154504459.26111)
    assert df_all["aqueous_outside"].sum() == pytest.approx(1861764062.6057913)
    assert df_all["gas_hazardous"].sum() == pytest.approx(1281418094.3408813)
    assert df_all["aqueous_hazardous"].sum() == pytest.approx(467152921.40563196)

    for zone in ZONES_SYNTHETIC_CASE:
        assert zone in df["zone"].unique()
    assert df[df["zone"] == "Amethyst"]["total"].sum() == pytest.approx(
        10654885273.532726
    )
    assert df[df["zone"] == "Ruby"]["total"].sum() == pytest.approx(2849437334.620893)
    assert df[df["zone"] == "Topaz"]["total"].sum() == pytest.approx(3748679729.6150413)


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
    # os.remove(output_path)

    assert len(df) == 66
    assert len(df["date"].unique()) == 11
    assert df["date"].min() == "2025-01-01"
    assert df["date"].max() == "2500-01-01"

    df_all = df[(df["zone"] == "all") & (df["region"] == "all")]
    assert df_all["total"].sum() == pytest.approx(22071879.550792538)
    assert df_all["total_gas"].sum() == pytest.approx(19033465.136888713)
    assert df_all["total_aqueous"].sum() == pytest.approx(3038414.4139038236)
    assert df_all["total_contained"].sum() == pytest.approx(11046790.6125475)
    assert df_all["total_outside"].sum() == pytest.approx(8799634.69523845)
    assert df_all["total_hazardous"].sum() == pytest.approx(2225454.2430065842)
    assert df_all["gas_contained"].sum() == pytest.approx(9851596.627884675)
    assert df_all["aqueous_contained"].sum() == pytest.approx(1195193.9846628257)
    assert df_all["gas_outside"].sum() == pytest.approx(7328181.264653008)
    assert df_all["aqueous_outside"].sum() == pytest.approx(1471453.4305854416)
    assert df_all["gas_hazardous"].sum() == pytest.approx(1853687.2443510273)
    assert df_all["aqueous_hazardous"].sum() == pytest.approx(371766.9986555566)

    for zone in ZONES_SYNTHETIC_CASE:
        assert zone in df["zone"].unique()
    assert df[df["zone"] == "Amethyst"]["total"].sum() == pytest.approx(
        14069030.980090078
    )
    assert df[df["zone"] == "Ruby"]["total"].sum() == pytest.approx(3532111.466066274)
    assert df[df["zone"] == "Topaz"]["total"].sum() == pytest.approx(4470737.104636182)


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
    # os.remove(output_path)

    assert len(df) == 66
    assert len(df["date"].unique()) == 11
    assert df["date"].min() == "2025-01-01"
    assert df["date"].max() == "2500-01-01"

    df_all = df[(df["zone"] == "all") & (df["region"] == "all")]
    assert df_all["total"].sum() == pytest.approx(2322814736)
    assert df_all["total_contained"].sum() == pytest.approx(270128250)
    assert df_all["total_outside"].sum() == pytest.approx(1817996448)
    assert df_all["total_hazardous"].sum() == pytest.approx(234690038)

    for zone in ZONES_SYNTHETIC_CASE:
        assert zone in df["zone"].unique()
    assert df[df["zone"] == "Amethyst"]["total"].sum() == pytest.approx(932432446)
    assert df[df["zone"] == "Ruby"]["total"].sum() == pytest.approx(440557772)
    assert df[df["zone"] == "Topaz"]["total"].sum() == pytest.approx(949824518)


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
    ]
    mocker.patch(
        "sys.argv",
        args,
    )
    main()

    output_path = str(main_path / "share" / "results" / "tables" / "plume_mass.csv")
    df = pandas.read_csv(output_path)
    os.remove(output_path)

    # assert len(df) == 66
    # assert len(df["date"].unique()) == 11
    # assert df["date"].min() == "2025-01-01"
    # assert df["date"].max() == "2500-01-01"
    #
    # df_all = df[(df["zone"] == "all") & (df["region"] == "all")]
    # assert df_all["total"].sum() == pytest.approx(17253002337.768658)
    # assert df_all["total_gas"].sum() == pytest.approx(13426268976.322)
    # assert df_all["total_aqueous"].sum() == pytest.approx(3826733361.4466577)
    # assert df_all["total_contained"].sum() == pytest.approx(8488162800.155235)
    # assert df_all["total_outside"].sum() == pytest.approx(7016268521.866909)
    # assert df_all["total_hazardous"].sum() == pytest.approx(1748571015.7465131)
    # assert df_all["gas_contained"].sum() == pytest.approx(6990346422.720001)
    # assert df_all["aqueous_contained"].sum() == pytest.approx(1497816377.435234)
    # assert df_all["gas_outside"].sum() == pytest.approx(5154504459.26111)
    # assert df_all["aqueous_outside"].sum() == pytest.approx(1861764062.6057913)
    # assert df_all["gas_hazardous"].sum() == pytest.approx(1281418094.3408813)
    # assert df_all["aqueous_hazardous"].sum() == pytest.approx(467152921.40563196)
    #
    # for zone in ZONES_SYNTHETIC_CASE:
    #     assert zone in df["zone"].unique()
    # assert df[df["zone"] == "Amethyst"]["total"].sum() == pytest.approx(
    #     10654885273.532726
    # )
    # assert df[df["zone"] == "Ruby"]["total"].sum() == pytest.approx(2849437334.620893)
    # assert df[df["zone"] == "Topaz"]["total"].sum() == pytest.approx(3748679729.6150413)


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
    ]
    mocker.patch(
        "sys.argv",
        args,
    )
    main()

    output_path = str(main_path / "share" / "results" / "tables" / "plume_mass.csv")
    df = pandas.read_csv(output_path)
    # os.remove(output_path)

    assert len(df) == 186
    assert len(df["date"].unique()) == 31

    df_all = df[(df["zone"] == "all") & (df["region"] == "all")]
    assert df_all["date"].min() == "2025-01-01"
    assert df_all["date"].max() == "2500-01-01"
    assert df_all["total"].sum() == pytest.approx(53558117131.70304)
    assert df_all["total_gas"].sum() == pytest.approx(42284432871.28494)
    assert df_all["total_aqueous"].sum() == pytest.approx(11273684260.41809)
    assert df_all["total_contained"].sum() == pytest.approx(26031030660.64489)
    assert df_all["total_outside"].sum() == pytest.approx(22401493179.59596)
    assert df_all["total_hazardous"].sum() == pytest.approx(5125593291.462185)
    assert df_all["gas_contained"].sum() == pytest.approx(21361675098.92482)
    assert df_all["aqueous_contained"].sum() == pytest.approx(4669355561.72007)
    assert df_all["gas_outside"].sum() == pytest.approx(16950990654.27981)
    assert df_all["aqueous_outside"].sum() == pytest.approx(5450502525.316155)
    assert df_all["gas_hazardous"].sum() == pytest.approx(3971767118.0803223)
    assert df_all["aqueous_hazardous"].sum() == pytest.approx(1153826173.3818629)

    for zone in ZONES_SYNTHETIC_CASE:
        assert zone in df["zone"].unique()
    assert df[df["zone"] == "Amethyst"]["total"].sum() == pytest.approx(
        37958705912.31001
    )
    assert df[df["zone"] == "Ruby"]["total"].sum() == pytest.approx(6356360232.242859)
    assert df[df["zone"] == "Topaz"]["total"].sum() == pytest.approx(9243050987.15018)


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
    # os.remove(output_path)

    assert len(df) == 186
    assert len(df["date"].unique()) == 31

    df_all = df[(df["zone"] == "all") & (df["region"] == "all")]
    assert df_all["date"].min() == "2025-01-01"
    assert df_all["date"].max() == "2500-01-01"
    assert df_all["total"].sum() == pytest.approx(75595526.20184366)
    assert df_all["total_gas"].sum() == pytest.approx(60009218.648827046)
    assert df_all["total_aqueous"].sum() == pytest.approx(15586307.55301664)
    assert df_all["total_contained"].sum() == pytest.approx(34604977.322922826)
    assert df_all["total_outside"].sum() == pytest.approx(33708138.97052885)
    assert df_all["total_hazardous"].sum() == pytest.approx(7282409.908391995)
    assert df_all["gas_contained"].sum() == pytest.approx(30127796.47406921)
    assert df_all["aqueous_contained"].sum() == pytest.approx(4477180.848853616)
    assert df_all["gas_outside"].sum() == pytest.approx(24168349.27021681)
    assert df_all["aqueous_outside"].sum() == pytest.approx(9539789.70031204)
    assert df_all["gas_hazardous"].sum() == pytest.approx(5713072.904541016)
    assert df_all["aqueous_hazardous"].sum() == pytest.approx(1569337.003850981)

    for zone in ZONES_SYNTHETIC_CASE:
        assert zone in df["zone"].unique()
    assert df[df["zone"] == "Amethyst"]["total"].sum() == pytest.approx(
        53224174.46785085
    )
    assert df[df["zone"] == "Ruby"]["total"].sum() == pytest.approx(8954581.057674848)
    assert df[df["zone"] == "Topaz"]["total"].sum() == pytest.approx(13416770.676317988)


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
    # os.remove(output_path)

    assert len(df) == 186
    assert len(df["date"].unique()) == 31

    df_all = df[(df["zone"] == "all") & (df["region"] == "all")]
    assert df_all["date"].min() == "2025-01-01"
    assert df_all["date"].max() == "2500-01-01"
    assert df_all["total"].sum() == pytest.approx(7837753267)
    assert df_all["total_contained"].sum() == pytest.approx(870413250)
    assert df_all["total_outside"].sum() == pytest.approx(6210951338)
    assert df_all["total_hazardous"].sum() == pytest.approx(756388679)

    for zone in ZONES_SYNTHETIC_CASE:
        assert zone in df["zone"].unique()
    assert df[df["zone"] == "Amethyst"]["total"].sum() == pytest.approx(3163356454)
    assert df[df["zone"] == "Ruby"]["total"].sum() == pytest.approx(1485640963)
    assert df[df["zone"] == "Topaz"]["total"].sum() == pytest.approx(3188755850)
