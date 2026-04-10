import os
import shutil
import tempfile
from pathlib import Path

import pytest
import xtgeo

from ccs_scripts.aggregate import grid3d_aggregate_map


def test_aggregated_map1():
    result = Path(__file__).absolute().parent / "aggregate1_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    cfg = "tests/yaml/config_aggregate1.yml"

    grid3d_aggregate_map.main(
        [
            "--config_aggregate",
            cfg,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    swat = xtgeo.surface_from_file(result / "all--max_swat--20030101.gri")
    assert swat.values.min() == pytest.approx(0.14292679727077484, abs=1e-8)
    shutil.rmtree(str(Path(result)))


def test_aggregated_map2():
    result = Path(__file__).absolute().parent / "aggregate2_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    cfg = "tests/yaml/config_aggregate2.yml"

    grid3d_aggregate_map.main(
        [
            "--config_aggregate",
            cfg,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    swat = xtgeo.surface_from_file(result / "all--min_swat--20030101.gri")
    assert swat.values.mean() == pytest.approx(0.7908786104444353, abs=1e-8)
    shutil.rmtree(str(Path(result)))


def test_aggregated_map3():
    result = Path(__file__).absolute().parent / "aggregate3_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    cfg = "tests/yaml/config_aggregate3.yml"

    grid3d_aggregate_map.main(
        [
            "--config_aggregate",
            cfg,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    poro = xtgeo.surface_from_file(result / "all--mean_poro.gri")
    assert poro.values.mean() == pytest.approx(0.1677586422488292, abs=1e-8)
    shutil.rmtree(str(Path(result)))


def test_aggregated_map4():
    result = Path(__file__).absolute().parent / "aggregate4_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    yml = "tests/yaml/config_aggregate4.yml"

    grid3d_aggregate_map.main(
        [
            "--config_aggregate",
            yml,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    swat = xtgeo.surface_from_file(result / "zone1--max_swat--20030101.gri")
    assert swat.values.max() == pytest.approx(1.0000962018966675, abs=1e-8)
    assert (result / "all--max_swat--20030101.gri").is_file()
    assert (result / "zone2--max_swat--20030101.gri").is_file()
    assert (result / "zone3--max_swat--20030101.gri").is_file()
    shutil.rmtree(str(Path(result)))


def test_aggregated_map5():
    result = Path(__file__).absolute().parent / "aggregate5_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    cfg = "tests/yaml/config_aggregate5.yml"

    grid3d_aggregate_map.main(
        [
            "--config_aggregate",
            cfg,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    poro = xtgeo.surface_from_file(result / "all--mean_poro.gri")
    assert poro.values.mean() == pytest.approx(0.1648792893163274, abs=1e-5)
    shutil.rmtree(str(Path(result)))


def test_aggregated_map6():
    result = Path(__file__).absolute().parent / "aggregate6_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    cfg = "tests/yaml/config_aggregate6.yml"

    grid3d_aggregate_map.main(
        [
            "--config_aggregate",
            cfg,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    gri_files = [p.stem for p in Path(result).glob("*.gri")]
    assert sorted(gri_files) == sorted(
        [
            "all--max_swat--19991201",
            "all--max_swat--20030101",
            "firstzone--max_swat--19991201",
            "firstzone--max_swat--20030101",
            "secondzone--max_swat--19991201",
            "secondzone--max_swat--20030101",
            "thirdzone--max_swat--19991201",
            "thirdzone--max_swat--20030101",
        ]
    )
    shutil.rmtree(str(Path(result)))


def test_aggregated_map7():
    result = Path(__file__).absolute().parent / "aggregate7_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    cfg = "tests/yaml/config_aggregate7.yml"

    grid3d_aggregate_map.main(
        [
            "--config_aggregate",
            cfg,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    gri_files = [p.stem for p in Path(result).glob("*.gri")]
    assert sorted(gri_files) == sorted(
        [
            "all--max_sgstrand--24000101",
            "all--max_sgstrand--25000101",
        ]
    )
    shutil.rmtree(str(Path(result)))


def test_aggregated_map8():
    result = Path(__file__).absolute().parent / "aggregate8_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))

    result_y = result / "weight-by-dz-yes"
    result_n = result / "weight-by-dz-no"
    result_y.mkdir(parents=True)
    result_n.mkdir(parents=True)

    cfg = "tests/yaml/config_aggregate8.yml"
    assert "weight_by_dz: yes" in Path(cfg).read_text()

    # Run with weight_by_dz: yes
    grid3d_aggregate_map.main(
        [
            "--config_aggregate",
            cfg,
            "--mapfolder",
            str(result_y),
            "--plotfolder",
            str(result_y),
        ]
    )

    # Run the exact same config but with weight_by_dz: no
    cfg_content = Path(cfg).read_text()
    cfg_content_no_weight = cfg_content.replace("weight_by_dz: yes", "weight_by_dz: no")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp_cfg:
        tmp_cfg.write(cfg_content_no_weight)
        tmp_cfg_path = tmp_cfg.name

    try:
        # Run with weight_by_dz: no
        grid3d_aggregate_map.main(
            [
                "--config_aggregate",
                tmp_cfg_path,
                "--mapfolder",
                str(result_n),
                "--plotfolder",
                str(result_n),
            ]
        )

        # Compare results - maps with dz-weighting should have higher values
        surf_with_dz = xtgeo.surface_from_file(result_y / "all--sum_permx.gri")
        surf_without_dz = xtgeo.surface_from_file(result_n / "all--sum_permx.gri")

        # Assert that dz-weighted values are higher than non-weighted
        assert surf_with_dz.values.mean() > surf_without_dz.values.mean()
        assert surf_with_dz.values.max() > surf_without_dz.values.max()
    finally:
        # Clean up temp file
        os.unlink(tmp_cfg_path)

    shutil.rmtree(str(Path(result)))
