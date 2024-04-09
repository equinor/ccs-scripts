import os
import shutil
from pathlib import Path

from resdata.resfile import FortIO, ResdataFile, openFortIO

from ccs_scripts.co2_mass_maps import co2_mass_maps


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
    new_unrst_file = str(
        Path(__file__).absolute().parent
        / "data"
        / "reek"
        / "eclipse"
        / "model"
        / "2_R001_REEK-0-mass-maps.UNRST"
    )
    shutil.copy(str(reek_unrstfile), new_unrst_file)
    with openFortIO(new_unrst_file, mode=FortIO.APPEND_MODE) as f:
        for y in AMFG:
            y.name = "AMFG"
            a = y.numpy_view()
            for i in range(0, len(a)):
                a[i] = a[i] * 0.02
            y.fwrite(f)
        for y in YMFG:
            y.name = "YMFG"
            a = y.numpy_view()
            for i in range(0, len(a)):
                a[i] = 0.99
            y.fwrite(f)
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


def test_co2_mass_maps_reek_grid():
    """
    Test CO2 containment code, with eclipse Reek data.
    Tests both mass and actual_volume calculations.
    """
    adapt_reek_grid_for_co2_mass_maps_test()
    result = str(Path(__file__).absolute().parent / "answers" / "mass_maps")
    co2_mass_maps.main(
        [
            "--config",
            str(
                Path(__file__).absolute().parent
                / "yaml"
                / "config_co2_mass_maps_reek.yml"
            ),
            "--mapfolder",
            str(result),
        ]
    )
    dissolved_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_maps"
        / "all--co2-mass-aqu-phase--20010801.gri"
    )
    free_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_maps"
        / "all--co2-mass-gas-phase--20010801.gri"
    )
    total_co2_file = (
        Path(__file__).absolute().parent
        / "answers"
        / "mass_maps"
        / "all--co2-mass-total--20010801.gri"
    )
    assert dissolved_co2_file.exists()
    assert free_co2_file.exists()
    assert total_co2_file.exists()
    shutil.rmtree(str(Path(__file__).absolute().parent / "answers" / "mass_maps"))
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
