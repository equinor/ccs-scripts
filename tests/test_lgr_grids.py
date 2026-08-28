import os
import shutil
from pathlib import Path

import numpy as np
import pandas
import pytest
import resfo
import xtgeo
from resdata.resfile import ResdataFile
from resdata.summary import Summary

from ccs_scripts.aggregate import grid3d_aggregate_map, grid3d_co2_mass_map
from ccs_scripts.aggregate._config import (
    AggregationMethod,
    CO2MassSettings,
    ComputeSettings,
    Input,
    Output,
    Property,
    RootConfig,
)
from ccs_scripts.co2_containment.co2_containment import main
from ccs_scripts.co2_containment.source_data import (
    LGR_PORV_OLD_PARENT_VALUE,
    _aggregate_lgr_porv_to_active_parent_cells,
)


@pytest.fixture
def lgr_data_dir():
    return Path(__file__).parent / "lgr-model"


@pytest.fixture
def lgr_co2_mass_config(lgr_data_dir, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return RootConfig(
        input=Input(
            grid=str(lgr_data_dir / "DEP_GAS_4.EGRID"),
        ),
        output=Output(
            mapfolder=str(output_dir),
        ),
        computesettings=ComputeSettings(
            aggregation=AggregationMethod.DISTRIBUTE,
            zone=False,
        ),
        co2_mass_settings=CO2MassSettings(
            unrst_source=str(lgr_data_dir / "DEP_GAS_4.UNRST"),
            init_source=str(lgr_data_dir / "DEP_GAS_4.INIT"),
            cirrus_info_file=str(lgr_data_dir / "DEP_GAS_4_INFO.csv"),
        ),
    )


@pytest.fixture
def lgr_aggregate_sgas_config(lgr_data_dir, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return RootConfig(
        input=Input(
            grid=str(lgr_data_dir / "DEP_GAS_4.EGRID"),
            properties=[
                Property(
                    source=str(lgr_data_dir / "DEP_GAS_4.UNRST"),
                    name="SGAS",
                )
            ],
        ),
        output=Output(
            mapfolder=str(output_dir),
        ),
        computesettings=ComputeSettings(
            aggregation=AggregationMethod.MEAN,
            zone=False,
        ),
    )


def test_mass_maps_with_lgr(lgr_data_dir, lgr_co2_mass_config):
    output_dir = Path(lgr_co2_mass_config.output.mapfolder)
    grid_output_dir = output_dir / "3d"
    lgr_co2_mass_config.output.gridfolder = str(grid_output_dir)

    grid3d_co2_mass_map.generate_co2_mass_maps(lgr_co2_mass_config)
    # 9 time stamps, 3 maps per timestamp:
    assert len(list(Path(output_dir).glob("*.gri"))) == 9 * 3
    expected_properties = {
        "co2_mass_dissolved_water_phase": "MASSDISW",
        "co2_mass_gas_phase": "MASS_GAS",
        "co2_mass_total": "MASS_TOT",
    }
    assert sorted(path.stem for path in grid_output_dir.glob("*.EGRID")) == ["co2_mass"]
    assert sorted(path.stem for path in grid_output_dir.glob("*.UNRST")) == ["co2_mass"]
    egrid = ResdataFile(str(grid_output_dir / "co2_mass.EGRID"))
    restart = ResdataFile(str(grid_output_dir / "co2_mass.UNRST"))
    assert len(egrid["GRIDHEAD"]) > 0
    assert len(restart["SEQNUM"]) == 9
    for keyword in expected_properties.values():
        assert len(restart[keyword]) == 9

    output_grid = xtgeo.grid_from_file(str(grid_output_dir / "co2_mass.EGRID"))
    output_properties = xtgeo.gridproperties_from_file(
        str(grid_output_dir / "co2_mass.UNRST"),
        names=list(expected_properties.values()),
        dates="all",
        grid=output_grid,
    )
    assert len(output_properties.props) == 9 * len(expected_properties)

    # In this Cirrus model, the following keywords are present:
    # - FSMDS (dissolved)
    # - FSMMO (mobile)
    # - FSMTR (trapped)
    # Their sum should equal FSMIP (total).
    # dict[(date, property)] -> value from the summary file for easy lookup:
    smry = Summary(str(lgr_data_dir / "DEP_GAS_4"))
    unsmry: dict[tuple[str, str], float] = {}
    for i, dt in enumerate(smry.report_dates):
        date_str = dt.strftime("%Y%m%d")
        for prop in ["FSMIP", "FSMDS", "FSMMO", "FSMTR"]:
            # Divide by 1000 for proper comparison
            unsmry[(date_str, prop)] = (
                smry.numpy_vector(prop, report_only=True)[i] / 1000
            )

    # Compare total amount of CO2 in total and dissolved maps to summary
    # values. We allow a 1% relative difference, which is somewhat arbitrary
    # but should be sufficient to catch major issues with the LGR handling.
    # TODO: look into comparing FSMMO as well
    total_gri_files = sorted(Path(output_dir).glob("all--*co2_mass_total--*.gri"))
    assert len(total_gri_files) == 9
    for gri_path in total_gri_files:
        date_str = gri_path.stem.split("--")[-1]
        surface = xtgeo.surface_from_file(str(gri_path))
        gri_total = float(np.ma.filled(surface.values, 0.0).sum())
        unsmry_total = unsmry[(date_str, "FSMIP")]
        assert gri_total == pytest.approx(unsmry_total, rel=0.01)

    dissolved_gri_files = sorted(
        Path(output_dir).glob("all--*co2_mass_dissolved_water_phase--*.gri")
    )
    assert len(dissolved_gri_files) == 9
    for gri_path in dissolved_gri_files:
        date_str = gri_path.stem.split("--")[-1]
        surface = xtgeo.surface_from_file(str(gri_path))
        gri_total = float(np.ma.filled(surface.values, 0.0).sum())
        fsmds_total = unsmry[(date_str, "FSMDS")]
        assert gri_total == pytest.approx(fsmds_total, rel=0.01)


def test_aggregate_maps_sgas_smooth_with_lgr(lgr_aggregate_sgas_config):
    """
    Verify that mean-SGAS aggregate maps produced from a grid containing LGR
    cells are smooth.  A non-smooth result (large jumps between adjacent map
    cells) indicates that the LGR section of the grid is being treated as
    inactive during aggregation, creating a hole where the refined cells are.
    """
    output_dir = Path(lgr_aggregate_sgas_config.output.mapfolder)

    grid3d_aggregate_map.generate_maps(
        lgr_aggregate_sgas_config.input,
        lgr_aggregate_sgas_config.zonation,
        lgr_aggregate_sgas_config.computesettings,
        lgr_aggregate_sgas_config.mapsettings,
        lgr_aggregate_sgas_config.output,
    )

    sgas_maps = sorted(output_dir.glob("all--mean_sgas--*.gri"))
    assert len(sgas_maps) == 9

    for gri_path in sgas_maps:
        surface = xtgeo.surface_from_file(str(gri_path))
        filled = np.ma.filled(surface.values, np.nan)
        max_adjacent_diff = max(
            np.nanmax(np.abs(np.diff(filled, axis=0))),
            np.nanmax(np.abs(np.diff(filled, axis=1))),
        )
        assert max_adjacent_diff < 0.1, (
            f"SGAS map {gri_path.name} is not smooth: "
            f"max adjacent cell difference = {max_adjacent_diff:.4f}. "
            "This likely means the LGR section is inactive during grid aggregation."
        )


def _get_lgr_case_paths():
    file_name = "DEP_GAS_4"
    main_path = Path(__file__).parents[1] / "tests" / "lgr-model"
    case_path = str(main_path / file_name)
    root_dir = ""
    output_dir = str(main_path / "share" / "results" / "tables")
    return (
        main_path,
        case_path,
        root_dir,
        output_dir,
    )


def _patch_init_porv_old_value_with_children_sum(
    grid_file: str, init_file: str, patched_init_file: str
) -> None:
    """
    Write a copy of init_file where every LGR parent cell with PORV=1.0
    is replaced by the sum of PORV of its active child cells
    """

    grid = xtgeo.grid_from_file(grid_file)
    init = xtgeo.gridproperties_from_file(init_file, grid=grid, names=["PORV"])
    active_cells = grid.actnum_array.astype(bool)
    nx, ny, _ = active_cells.shape

    porv_prop = init.get_prop_by_name("PORV")
    porv_vals = porv_prop.values[active_cells].data

    fixed_active_porv, _ = _aggregate_lgr_porv_to_active_parent_cells(
        grid_file, init_file, active_cells, porv_vals
    )

    # Mapping the fixed, active-cells-only PORV back to full EGRID cell order
    # same as in source_data.py
    active_ijk = np.argwhere(active_cells)
    egrid_indices = (
        active_ijk[:, 0] + nx * active_ijk[:, 1] + nx * ny * active_ijk[:, 2]
    )

    porv_seen = 0
    patched_entries = []
    for keyword, array in resfo.read(init_file):
        if keyword.strip() == "PORV":
            if porv_seen == 0:  # Only fix PORV in the main grid
                fixed_full = np.asarray(array, dtype=array.dtype).copy()
                fixed_full[egrid_indices] = fixed_active_porv.astype(array.dtype)
                array = fixed_full
            porv_seen += 1
        patched_entries.append((keyword, array))

    resfo.write(patched_init_file, patched_entries)

    fixed_init = xtgeo.gridproperties_from_file(
        patched_init_file, grid=grid, names=["PORV"]
    )
    fixed_porv = fixed_init.get_prop_by_name("PORV")
    fixed_vals = fixed_porv.values[active_cells].data
    assert not np.any(fixed_vals == LGR_PORV_OLD_PARENT_VALUE)


def test_lgr_co2_amount_old_and_new_cirrus(mocker):
    """
    Test CO2 containment for cases with LGRs in two scenarios:
    1 - Old Cirrus cases where PORV=1 is replaced by aggregation of child cells
    2 - New Cirrus cases where PORV is assumed right, still verified internally
    """
    (
        main_path,
        case_path,
        root_dir,
        out_dir,
    ) = _get_lgr_case_paths()
    grid_file = str(case_path) + ".EGRID"
    init_file = str(case_path) + ".INIT"
    cirrus_info_file = str(case_path) + "_INFO.csv"
    patched_init_file = str(case_path) + "_new_cirrus.INIT"
    _patch_init_porv_old_value_with_children_sum(
        grid_file, init_file, patched_init_file
    )
    args = [
        "sys.argv",
        case_path,
        "mass",
        "--root_dir",
        root_dir,
        "--out_dir",
        out_dir,
        "--cirrus_info_file",
        cirrus_info_file,
    ]
    os.makedirs(out_dir, exist_ok=True)
    try:
        mocker.patch("sys.argv", args + ["--init", init_file])
        main()
        df_old = pandas.read_csv(Path(out_dir) / "plume_mass.csv")

        mocker.patch("sys.argv", args + ["--init", patched_init_file])
        main()
        df_new = pandas.read_csv(Path(out_dir) / "plume_mass.csv")
    finally:
        shutil.rmtree(main_path / "share", ignore_errors=True)
        os.remove(patched_init_file)

    answer_file = str(
        Path(__file__).parents[0]
        / "answers"
        / "containment"
        / "plume_mass_lgr_cirrus.csv"
    )
    df_answer = pandas.read_csv(answer_file)

    df_old = df_old.sort_values("date").reset_index(drop=True)
    df_new = df_new.sort_values("date").reset_index(drop=True)
    df_answer = df_answer.sort_values("date").reset_index(drop=True)
    pandas.testing.assert_frame_equal(df_old, df_answer)
    pandas.testing.assert_frame_equal(df_new, df_answer)
