import os
import shutil
from pathlib import Path

from resdata.resfile import FortIO, ResdataFile, openFortIO

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
    properties = ResdataFile(str(reek_unrstfile))
    SGAS = properties["SGAS"]
    AMFG = []
    YMFG = []
    DGAS = []
    DWAT = []
    for x in SGAS:
        amfg = x.copy()
        amfg.name = "AMFG"
        amfg.numpy_view()[:] *= 0.02
        AMFG.append(amfg)

        ymfg = x.copy()
        ymfg.name = "YMFG"
        ymfg.numpy_view()[:] = 0.99
        YMFG.append(ymfg)

        dgas = x.copy()
        dgas.name = "DGAS"
        dgas.numpy_view()[:] = 100
        DGAS.append(dgas)

        dwat = x.copy()
        dwat.name = "DWAT"
        dwat.numpy_view()[:] = 1000
        DWAT.append(dwat)

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
    seqnum_count = 0
    with openFortIO(new_unrst_file, mode=FortIO.WRITE_MODE) as f:
        for i in range(len(properties)):
            kw = properties[i]
            kw.fwrite(f)
            if kw.name == "SEQNUM":
                AMFG[seqnum_count].fwrite(f)
                YMFG[seqnum_count].fwrite(f)
                DGAS[seqnum_count].fwrite(f)
                DWAT[seqnum_count].fwrite(f)
                seqnum_count += 1


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
