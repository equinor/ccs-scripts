from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import scipy.ndimage
import shapely.geometry
import xtgeo

from ccs_scripts.co2_containment.co2_calculation import (
    RELEVANT_PROPERTIES,
    CalculationType,
    SourceData,
    _calculate_co2_data_from_source_data,
    _extract_source_data,
)
from ccs_scripts.co2_containment.co2_containment import (
    calculate_from_co2_data,
    extract_amount,
    sort_and_replace_nones,
)

zone_info = {
    "source": None,
    "zranges": None,
    "int_to_zone": None,
}
region_info = {
    "source": None,
    "int_to_region": None,
    "property_name": None,
}


def _random_prop(
    dims: Tuple,
    rng,
    low: float,
    high: float,
):
    """
    Create random property
    """
    white = rng.normal(size=dims)
    smooth = scipy.ndimage.gaussian_filter(white, max(dims) / 10)
    values = smooth - np.min(smooth)
    values /= np.max(values)
    values *= high
    values += low
    return values.flatten()


def _xy_and_volume(grid: xtgeo.Grid):
    """
    Get xy and volume
    """
    xyz = grid.get_xyz()
    vol = grid.get_bulk_volume().values1d.compressed()
    return xyz[0].values1d.compressed(), xyz[1].values1d.compressed(), vol


def _get_dummy_co2_masses():
    """
    Create dummy co2 mass data
    """
    dims = (11, 13, 7)
    dummy_co2_grid = xtgeo.create_box_grid(dims)

    n_time_steps = 10
    # pylint: disable-next=no-member
    rng = np.random.RandomState(123)
    x_coord, y_coord, vol = _xy_and_volume(dummy_co2_grid)
    dates = [str(2020 + i) for i in range(n_time_steps)]
    source_data = SourceData(
        x_coord,
        y_coord,
        PORV={date: _random_prop(dims, rng, 0.1, 0.3) for date in dates},
        VOL=vol,
        DATES=dates,
        SWAT={date: _random_prop(dims, rng, 0.05, 0.6) for date in dates},
        DWAT={date: _random_prop(dims, rng, 950, 1050) for date in dates},
        SGAS={date: _random_prop(dims, rng, 0.05, 0.6) for date in dates},
        DGAS={date: _random_prop(dims, rng, 700, 850) for date in dates},
        AMFG={date: _random_prop(dims, rng, 0.001, 0.01) for date in dates},
        YMFG={date: _random_prop(dims, rng, 0.001, 0.01) for date in dates},
    )
    return _calculate_co2_data_from_source_data(source_data, CalculationType.MASS)


def _calc_and_compare(poly, masses, poly_hazardous=None):
    totals = {m.date: np.sum(m.total_mass()) for m in masses.data_list}
    contained = calculate_from_co2_data(
        co2_data=masses,
        containment_polygon=poly,
        hazardous_polygon=poly_hazardous,
        calc_type_input="mass",
        zone_info=zone_info,
        region_info=region_info,
    )
    sort_and_replace_nones(contained)
    total_values = contained[
        (contained["phase"] == "total") & (contained["containment"] == "total")
    ]["amount"]
    difference = np.sum([x - y for x, y in zip(total_values, list(totals.values()))])
    assert difference == pytest.approx(0.0, abs=1e-8)
    return contained


def test_single_poly_co2_containment():
    """
    Test CO2 containment code, single polygon
    """
    dummy_co2_masses = _get_dummy_co2_masses()
    assert len(dummy_co2_masses.data_list) == 10
    poly = shapely.geometry.Polygon(
        [
            [7.1, 7.0],
            [9.1, 9.0],
            [7.1, 11.0],
            [5.1, 9.0],
            [7.1, 7.0],
        ]
    )
    table = _calc_and_compare(poly, dummy_co2_masses)
    assert extract_amount(table, "contained", "gas") == pytest.approx(90.262207)
    assert extract_amount(
        table,
        "contained",
        "aqueous",
    ) == pytest.approx(172.72921760648467)
    assert extract_amount(table, "hazardous", "gas") == pytest.approx(0.0)
    assert extract_amount(
        table,
        "hazardous",
        "aqueous",
    ) == pytest.approx(0.0)


def test_multi_poly_co2_containment():
    """ "
    Test CO2 containment code, muliple polygons
    """
    dummy_co2_masses = _get_dummy_co2_masses()
    poly = shapely.geometry.MultiPolygon(
        [
            shapely.geometry.Polygon(
                [
                    [7.1, 7.0],
                    [9.1, 9.0],
                    [7.1, 11.0],
                    [5.1, 9.0],
                    [7.1, 7.0],
                ]
            ),
            shapely.geometry.Polygon(
                [
                    [1.0, 1.0],
                    [3.0, 1.0],
                    [3.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 1.0],
                ]
            ),
        ]
    )
    table = _calc_and_compare(poly, dummy_co2_masses)
    assert extract_amount(
        table,
        "contained",
        "gas",
    ) == pytest.approx(123.70267352027123)
    assert extract_amount(
        table,
        "contained",
        "aqueous",
    ) == pytest.approx(252.79970312163525)
    assert extract_amount(table, "hazardous", "gas") == pytest.approx(0.0)
    assert extract_amount(
        table,
        "hazardous",
        "aqueous",
    ) == pytest.approx(0.0)


def test_hazardous_poly_co2_containment():
    """ "
    Test CO2 containment code, with hazardous polygon
    """
    dummy_co2_masses = _get_dummy_co2_masses()
    assert len(dummy_co2_masses.data_list) == 10
    poly = shapely.geometry.Polygon(
        [
            [7.1, 7.0],
            [9.1, 9.0],
            [7.1, 11.0],
            [5.1, 9.0],
            [7.1, 7.0],
        ]
    )
    poly_hazardous = shapely.geometry.Polygon(
        [
            [9.1, 9.0],
            [9.1, 11.0],
            [7.1, 11.0],
            [9.1, 9.0],
        ]
    )
    table = _calc_and_compare(poly, dummy_co2_masses, poly_hazardous)
    assert extract_amount(table, "contained", "gas") == pytest.approx(90.262207)
    assert extract_amount(
        table,
        "contained",
        "aqueous",
    ) == pytest.approx(172.72921760648467)
    assert extract_amount(
        table,
        "hazardous",
        "gas",
    ) == pytest.approx(12.687891108274542)
    assert extract_amount(
        table,
        "hazardous",
        "aqueous",
    ) == pytest.approx(20.33893251315071)


def test_reek_grid():
    """
    Test CO2 containment code, with eclipse Reek data.
    Tests both mass and actual_volume calculations.
    Additionally, tests mass with residual_trapping
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
        calc_type_input="mass",
        zone_info=zone_info,
        region_info=region_info,
    )
    sort_and_replace_nones(table)
    cs = ["total"] * 3 + ["contained"] + ["hazardous"] * 2
    ps = ["total", "gas", "aqueous", "gas", "total", "gas"]
    amounts = [
        696171.20388324,
        7650.233009712884,
        688520.9708735272,
        115.98058252427084,
        10282.11650485436,
        112.99029126213496,
    ]
    for c, p, amount in zip(cs, ps, amounts):
        assert extract_amount(table, c, p, 0) == pytest.approx(amount)

    volumes = _calculate_co2_data_from_source_data(
        source_data,
        CalculationType.ACTUAL_VOLUME,
    )
    table2 = calculate_from_co2_data(
        co2_data=volumes,
        containment_polygon=reek_poly,
        hazardous_polygon=reek_poly_hazardous,
        calc_type_input="actual_volume",
        zone_info=zone_info,
        region_info=region_info,
    )
    sort_and_replace_nones(table2)
    amounts2 = [
        1018.524203883313,
        330.0032330095245,
        688.5209708737885,
        5.002980582524296,
        15.043116504854423,
        4.873990291262155,
    ]
    for c, p, amount in zip(cs, ps, amounts2):
        assert extract_amount(table2, c, p, 0) == pytest.approx(amount)

    source_data_with_trapping = SourceData(
        x_coord,
        y_coord,
        PORV={"2042": np.ones_like(poro) * 0.1},
        VOL=vol,
        DATES=["2042"],
        SWAT={"2042": np.ones_like(poro) * 0.1},
        DWAT={"2042": np.ones_like(poro) * 1000.0},
        SGAS={"2042": np.ones_like(poro) * 0.1},
        SGSTRAND={"2042": np.ones_like(poro) * 0.06},
        DGAS={"2042": np.ones_like(poro) * 100.0},
        AMFG={"2042": np.ones_like(poro) * 0.1},
        YMFG={"2042": np.ones_like(poro) * 0.1},
    )

    masses_with_trapping = _calculate_co2_data_from_source_data(
        source_data_with_trapping, CalculationType.MASS
    )
    table3 = calculate_from_co2_data(
        co2_data=masses_with_trapping,
        containment_polygon=reek_poly,
        hazardous_polygon=reek_poly_hazardous,
        calc_type_input="mass",
        zone_info=zone_info,
        region_info=region_info,
        residual_trapping=True,
    )
    sort_and_replace_nones(table3)
    cs3 = ["total"] * 4 + ["contained"] * 2 + ["hazardous"] * 3
    gas_part = ["trapped_gas", "free_gas"]
    ps3 = ["total"] + gas_part + ["aqueous"] + gas_part + ["total"] + gas_part
    amounts3 = [
        696171.20388324,
        4590.13980582773,
        3060.093203885154,
        688520.9708735272,
        69.58834951456248,
        46.39223300970832,
        10282.11650485436,
        67.79417475728094,
        45.19611650485396,
    ]
    for c, p, amount in zip(cs3, ps3, amounts3):
        assert extract_amount(table3, c, p, 0) == pytest.approx(amount)


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
            RELEVANT_PROPERTIES,
            zone_info,
            region_info,
            str(reek_initfile),
        )


def test_calculation_type():
    """Test invalid calculation type exception handling"""
    with pytest.raises(ValueError):
        CalculationType.check_for_key("mass")
