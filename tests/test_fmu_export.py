from pathlib import Path

import pytest
import yaml

from ccs_scripts.aggregate import grid3d_aggregate_map
from ccs_scripts.aggregate._config import (
    ComputeSettings,
    Input,
    MapOutputFormat,
    MapSettings,
    Output,
    Property,
    Zonation,
)

# Schema-valid GlobalConfiguration (masterdata.smda, model, access.asset) -
# fmu-dataio auto-discovers this by path convention
# (fmuconfig/output/global_variables.yml, two levels above the runpath).
_GLOBAL_VARIABLES = """
model:
  name: CCS-Sandbox
  revision: "0.0.1"

masterdata:
  smda:
    coordinate_system:
      identifier: ST_WGS84_UTM37N_P32637
      uuid: b6b5997a-84bf-44fc-a6e3-73d8d93e8f4f
    country:
      - identifier: Norway
        uuid: 63f46c9f-f83c-4632-bf0f-23b5b4e18a19
    discovery:
      - short_identifier: CcsSandboxDiscovery
        uuid: 33cd952b-a07d-4ebd-9bad-01cf2da4f1bb
    field:
      - identifier: CcsSandboxField
        uuid: ce3de93f-5282-44f7-bd20-4a09982e92e6
    stratigraphic_column:
      identifier: CCS_SANDBOX_2026
      uuid: ce3de93f-5282-44f7-bd20-4a09982e92e6

access:
  asset:
    name: CCS-Sandbox
  classification: internal
"""

# Stand-in for the file fmu-dataio's own WF_CREATE_CASE_METADATA ERT workflow
# writes at PRE_SIMULATION. Without this file, ExportData exports data fine but
# silently produces no `fmu:` metadata block (no error) - see HANDOFF.md.
_FMU_CASE_METADATA = """
class: case
masterdata:
  smda:
    coordinate_system:
      identifier: ST_WGS84_UTM37N_P32637
      uuid: b6b5997a-84bf-44fc-a6e3-73d8d93e8f4f
    country:
      - identifier: Norway
        uuid: 63f46c9f-f83c-4632-bf0f-23b5b4e18a19
    discovery:
      - short_identifier: CcsSandboxDiscovery
        uuid: 33cd952b-a07d-4ebd-9bad-01cf2da4f1bb
    field:
      - identifier: CcsSandboxField
        uuid: ce3de93f-5282-44f7-bd20-4a09982e92e6
    stratigraphic_column:
      identifier: CCS_SANDBOX_2026
      uuid: ce3de93f-5282-44f7-bd20-4a09982e92e6
tracklog:
  - datetime: "2026-08-20T00:00:00Z"
    event: created
    user:
      id: pytest
source: fmu
version: "0.24.0"
fmu:
  case:
    name: ccs-scripts-fmu-export-test
    user:
      id: pytest
    uuid: d01f2fd2-3dbe-4b2a-a7ae-f12d6fd2cbab
  model:
    name: CCS-Sandbox
    revision: "0.0.1"
access:
  asset:
    name: CCS-Sandbox
  classification: internal
"""


@pytest.fixture
def fmu_case_runpath(tmp_path):
    """
    Build a minimal but real FMU case directory tree, shaped exactly like what
    ERT produces: <case>/realization-0/iter-0 as the runpath, with
    fmuconfig/output/global_variables.yml and share/metadata/fmu_case.yml at
    the case root. Ported from the hand-built ERT case validated in
    ~/Documents/ccs-fmu-test/.
    """
    case_dir = tmp_path / "case"
    runpath = case_dir / "realization-0" / "iter-0"
    runpath.mkdir(parents=True)

    global_config_dir = case_dir / "fmuconfig" / "output"
    global_config_dir.mkdir(parents=True)
    (global_config_dir / "global_variables.yml").write_text(_GLOBAL_VARIABLES)

    case_metadata_dir = case_dir / "share" / "metadata"
    case_metadata_dir.mkdir(parents=True)
    (case_metadata_dir / "fmu_case.yml").write_text(_FMU_CASE_METADATA)

    return runpath


def test_real_aggregate_map_exports_with_fmu_metadata(fmu_case_runpath, monkeypatch):
    """
    A real ccs-scripts aggregate map (REEK SWAT, not a synthetic dummy surface)
    exported with output_format=fmu-dataio, run from inside a genuine
    ERT-shaped runpath, must produce a .gri file plus valid FMU metadata with a
    populated `fmu:` block (realization/case identity) - not just a metadata
    file with masterdata/access and nothing else, which is what you get
    without the case metadata / ERT env vars set up correctly.
    """
    repo_root = Path(__file__).resolve().parents[1]
    grid_file = str(repo_root / "tests" / "data" / "reek_3d_maps" / "REEK.EGRID")
    unrst_file = str(repo_root / "tests" / "data" / "reek_3d_maps" / "REEK.UNRST")

    # Mimics the env vars ERT itself sets for a forward model step
    # (fmu.dataio._runcontext.FMUEnvironment.from_env).
    monkeypatch.setenv("_ERT_RUNPATH", str(fmu_case_runpath))
    monkeypatch.setenv("_ERT_REALIZATION_NUMBER", "0")
    monkeypatch.setenv("_ERT_ITERATION_NUMBER", "0")
    monkeypatch.chdir(fmu_case_runpath)

    output = Output(
        mapfolder="unused-in-fmu-dataio-mode",
        output_format=MapOutputFormat.FMU_DATAIO,
    )
    input_ = Input(
        grid=grid_file,
        properties=[Property(source=unrst_file, name="SWAT", lower_threshold=1e-12)],
        dates=["20030101"],
    )

    grid3d_aggregate_map.generate_maps(
        input_,
        Zonation(),
        ComputeSettings(zone=False),
        MapSettings(),
        output,
    )

    maps_dir = fmu_case_runpath / "share" / "results" / "maps"
    exported = list(maps_dir.glob("*.gri"))
    assert len(exported) == 1, f"expected exactly one exported surface, found {exported}"

    metadata_files = list(maps_dir.glob(".*.gri.yml"))
    assert len(metadata_files) == 1, f"expected exactly one metadata file, found {metadata_files}"
    metadata = yaml.safe_load(metadata_files[0].read_text())

    assert metadata["data"]["content"] == "property"
    assert metadata["data"]["name"] == "max-swat"
    assert metadata["fmu"]["realization"]["id"] == 0
    assert metadata["fmu"]["realization"]["name"] == "realization-0"
    assert metadata["fmu"]["iteration"]["id"] == 0
    assert metadata["fmu"]["case"]["name"] == "ccs-scripts-fmu-export-test"
