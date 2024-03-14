from pathlib import Path
import numpy as np
import pytest
import xtgeo
import shutil
from resdata.resfile import ResdataFile,ResdataKW,openFortIO, FortIO
from ccs_scripts.co2_mass_maps import grid3d_co2_mass

def adapt_reek_grid_for_co2_mass_maps_test():
    """
    Adds the necessary properties to reek grid to make it usable for
    test_co2_mass_maps_reek_grid
    """
    reek_unrstfile = (
            Path(__file__).absolute().parent
            / "data"
            / "reek"
            / "eclipse"
            / "model"
            / "2_R001_REEK-0.UNRST"
    )
    properties = ResdataFile(str(reek_unrstfile))
    SGAS = properties["SGAS"]
    AMFG = []
    YMFG = []
    DGAS = []
    DWAT = []
    for x in SGAS:
        AMFG.append(x.copy())
        YMFG.append(x.copy())
        DGAS.append(x.copy())
        DWAT.append(x.copy())
    new_unrst_file = str(Path(__file__).absolute().parent
            / "data"
            / "reek"
            / "eclipse"
            / "model"
            / "2_R001_REEK-0-mass-maps.UNRST")
    shutil.copy(str(reek_unrstfile),new_unrst_file)
    with openFortIO(
            new_unrst_file,
            mode=FortIO.APPEND_MODE) as f:
        for y in AMFG:
            y.name = "AMFG"
            a = y.numpy_view()
            for i in range(0, len(a)):
                a[i] = a[i] * 0.02
            y.fwrite(f)
        for y in YMFG:
            a = y.numpy_view()
            for i in range(0, len(a)):
                a[i] = 0.99
        for y in DGAS:
            y.name = "DGAS"
            a = y.numpy_view()
            for i in range(0, len(a)):
                a[i] = 100
            y.fwrite(f)
        for y in DWAT:
            y.name = "DWAT"
            a = y.numpy_view()
            for i in range(0, len(a)):
                a[i] = 1000
            y.fwrite(f)

def test_co2_mass_maps_reek_grid(datatree):
    """
        Test CO2 containment code, with eclipse Reek data.
        Tests both mass and actual_volume calculations.
    """
    adapt_reek_grid_for_co2_mass_maps_test()
    result = datatree / "co2_mass_maps"
    result.mkdir(parents = True)
    grid3d_co2_mass.main(
        [
            "--config",
            "tests/yaml/config_co2_mass_maps_reek.yml",
            "--mapfolder",
            str(result)
        ]
    )
    #assert (result / "").is_file()
    #Pending: assert comparing result of co2containmentmass vs total in mass maps


def test_reek_grid():
    """
    Test CO2 containment code, with eclipse Reek data.
    Tests both mass and actual_volume calculations.
    """
    reek_gridfile = (
        Path(__file__).absolute().parent
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0.EGRID"
    )
    reek_poly = shapely.geometry.Polygon(
        [
            [461339, 5932377],
            [461339 + 1000, 5932377],
            [461339 + 1000, 5932377 + 1000],
            [461339, 5932377 + 1000],
        ]
    )
    reek_poly_hazardous = shapely.geometry.Polygon(
        [
            [461339 + 1000, 5932377],
            [461339 + 2000, 5932377],
            [461339 + 2000, 5932377 + 1000],
            [461339 + 1000, 5932377 + 1000],
            [461339 + 1000, 5932377],
        ]
    )
    grid = xtgeo.grid_from_file(reek_gridfile)
    poro = xtgeo.gridproperty_from_file(
        reek_gridfile.with_suffix(".INIT"), name="PORO", grid=grid
    ).values1d.compressed()
    x_coord, y_coord, vol = _xy_and_volume(grid)
    source_data = SourceData(
        x_coord,
        y_coord,
        PORV={"2042": np.ones_like(poro) * 0.1},
        VOL=vol,
        DATES=["2042"],
        SWAT={"2042": np.ones_like(poro) * 0.1},
        DWAT={"2042": np.ones_like(poro) * 1000.0},
        SGAS={"2042": np.ones_like(poro) * 0.1},
        DGAS={"2042": np.ones_like(poro) * 100.0},
        AMFG={"2042": np.ones_like(poro) * 0.1},
        YMFG={"2042": np.ones_like(poro) * 0.1},
    )
    masses = _calculate_co2_data_from_source_data(source_data, CalculationType.MASS)
    table = calculate_from_co2_data(
        co2_data=masses,
        containment_polygon=reek_poly,
        hazardous_polygon=reek_poly_hazardous,
        compact=False,
        calc_type_input="mass",
        zone_info=zone_info,
        region_info=region_info,
    )
    assert table.total.values[0] == pytest.approx(696171.20388324)
    assert table.total_gas.values[0] == pytest.approx(7650.233009712884)
    assert table.total_aqueous.values[0] == pytest.approx(688520.9708735272)
    assert table.gas_contained.values[0] == pytest.approx(115.98058252427084)
    assert table.total_hazardous.values[0] == pytest.approx(10282.11650485436)
    assert table.gas_hazardous.values[0] == pytest.approx(112.99029126213496)

    volumes = _calculate_co2_data_from_source_data(
        source_data,
        CalculationType.ACTUAL_VOLUME,
    )
    table2 = calculate_from_co2_data(
        co2_data=volumes,
        containment_polygon=reek_poly,
        hazardous_polygon=reek_poly_hazardous,
        compact=False,
        calc_type_input="actual_volume",
        zone_info=zone_info,
        region_info=region_info,
    )
    assert table2.total.values[0] == pytest.approx(1018.524203883313)
    assert table2.total_gas.values[0] == pytest.approx(330.0032330095245)
    assert table2.total_aqueous.values[0] == pytest.approx(688.5209708737885)
    assert table2.gas_contained.values[0] == pytest.approx(5.002980582524296)
    assert table2.total_hazardous.values[0] == pytest.approx(15.043116504854423)
    assert table2.gas_hazardous.values[0] == pytest.approx(4.873990291262155)


def test_reek_grid_extract_source_data():
    """
    Test CO2 containment code, with eclipse Reek data.
    Test extracting source data. Example does not have the
    required properties, so should get a RuntimeError
    """
    reek_gridfile = (
        Path(__file__).absolute().parent
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0.EGRID"
    )
    reek_unrstfile = (
        Path(__file__).absolute().parent
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0.UNRST"
    )
    reek_initfile = (
        Path(__file__).absolute().parent
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0.INIT"
    )
    with pytest.raises(RuntimeError):
        _extract_source_data(
            str(reek_gridfile),
            str(reek_unrstfile),
            PROPERTIES_TO_EXTRACT,
            zone_info,
            region_info,
            str(reek_initfile),
        )


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
