import os
import shutil
from pathlib import Path

import numpy as np
import resfo

from ccs_scripts.aggregate import grid3d_co2_mass_map


def adapt_reek_grid_for_co2_mass_map_test():
    """
    Adds the necessary properties to reek grid to make it usable for
    test_co2_mass_map_reek_grid
    """
    reek_unrstfile = (
        Path(__file__).absolute().parent
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0.UNRST"
    )
    records = resfo.read(reek_unrstfile)
    sgas_values = [
        np.asarray(values) for keyword, values in records if keyword.strip() == "SGAS"
    ]

    patched_records = []
    report_index = 0

    for keyword, values in records:
        patched_records.append((keyword, values))

        if keyword.strip() == "SEQNUM":
            sgas = sgas_values[report_index]
            patched_records.extend(
                [
                    ("AMFG    ", sgas * np.float32(0.02)),
                    ("YMFG    ", np.full_like(sgas, 0.99)),
                    ("DGAS    ", np.full_like(sgas, 100)),
                    ("DWAT    ", np.full_like(sgas, 1000)),
                ]
            )
            report_index += 1

    # The auxilliary properties needs to be written to the correct seqnum section
    # of the file, so we re-write the entire unrst file, and inject the properties
    # at the correct place.
    new_unrst_file = str(
        Path(__file__).absolute().parent
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0-mass-maps.UNRST"
    )
    resfo.write(new_unrst_file, patched_records)


def test_co2_mass_map_reek_grid():
    """
    Test CO2 mass maps generation, with eclipse Reek data
    """
    adapt_reek_grid_for_co2_mass_map_test()
    result = str(Path(__file__).absolute().parent / "answers" / "mass_map")
    if not os.path.exists(result):
        os.makedirs(result)
    grid3d_co2_mass_map.main(
        [
            "--config_co2_mass_map",
            str(
                Path(__file__).absolute().parent
                / "yaml"
                / "config_co2_mass_map_reek.yml"
            ),
            "--mapfolder",
            str(result),
            "--gridfolder",
            f"{str(result)}/3d",
        ]
    )
    dissolved_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_map"
        / "all--co2_mass_dissolved_water_phase--20010801.gri"
    )
    free_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_map"
        / "all--co2_mass_gas_phase--20010801.gri"
    )
    total_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_map"
        / "all--co2_mass_total--20010801.gri"
    )
    assert free_co2_file.exists()
    assert dissolved_co2_file.exists()
    assert total_co2_file.exists()
    shutil.rmtree(str(Path(__file__).absolute().parent / "answers" / "mass_map"))
    os.remove(
        str(
            Path(__file__).absolute().parent
            / "data"
            / "reek"
            / "eclipse"
            / "model"
            / "2_R001_REEK-0-mass-maps.UNRST"
        )
    )


def test_co2_mass_map_residual_trapping_cirrus():
    """
    Test CO2 mass maps, with synthetic_case cirrus data
    """
    result = str(Path(__file__).absolute().parent / "answers" / "mass_map")
    if not os.path.exists(result):
        os.makedirs(result)

    grid3d_co2_mass_map.main(
        [
            "--config_co2_mass_map",
            str(
                Path(__file__).absolute().parent
                / "yaml"
                / "config_co2_mass_map_cirrus.yml"
            ),
            "--mapfolder",
            str(result),
            "--gridfolder",
            f"{str(result)}/3d",
        ]
    )
    free_gas_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_map"
        / "all--co2_mass_free_gas_phase--23000101.gri"
    )
    trapped_gas_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_map"
        / "all--co2_mass_trapped_gas_phase--23000101.gri"
    )
    total_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_map"
        / "all--co2_mass_total--23000101.gri"
    )
    assert free_gas_co2_file.exists()
    assert trapped_gas_co2_file.exists()
    assert not total_co2_file.exists()
    shutil.rmtree(str(Path(__file__).absolute().parent / "answers" / "mass_map"))
