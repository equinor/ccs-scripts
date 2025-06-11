import shutil
from typing import Any, Dict

from ert import (
    ForwardModelStepDocumentation,
    ForwardModelStepPlugin,
)
from ert import plugin as ert_plugin

_CATEGORY = "modelling.reservoir"


class Co2ContainmentStep(ForwardModelStepPlugin):
    """Forward model step for CO2 containment."""

    def __init__(self):
        super().__init__(
            name="CO2_CONTAINMENT",
            command=Co2ContainmentStep._commands(),
            default_mapping=Co2ContainmentStep._default_mapping(),
        )

    @staticmethod
    def _default_mapping() -> Dict[str, Any]:
        return {
            "<ROOT_DIR>": -1,
            "<OUT_DIR>": -1,
            "<CONTAINMENT_POLYGON>": -1,
            "<HAZARDOUS_POLYGON>": -1,
            "<ZONEFILE>": -1,
            "<REGIONFILE>": -1,
            "<REGION_PROPERTY>": -1,
            "<EGRID>": -1,
            "<UNRST>": -1,
            "<INIT>": -1,
            "<NO_LOGGING>": -1,
            "<DEBUG>": -1,
            "<RESIDUAL_TRAPPING>": -1,
            "<READABLE_OUTPUT>": -1,
            "<CONFIG_PLUME_TRACKING>": "",
            "<GAS_MOLAR_MASS>": -1,
        }

    @staticmethod
    def _commands():
        return [
            shutil.which("co2_containment"),
            "<CASE>",
            "<CALC_TYPE_INPUT>",
            "--root_dir",
            "<ROOT_DIR>",
            "--out_dir",
            "<OUT_DIR>",
            "--containment_polygon",
            "<CONTAINMENT_POLYGON>",
            "--hazardous_polygon",
            "<HAZARDOUS_POLYGON>",
            "--zonefile",
            "<ZONEFILE>",
            "--regionfile",
            "<REGIONFILE>",
            "--region_property",
            "<REGION_PROPERTY>",
            "--egrid",
            "<EGRID>",
            "--unrst",
            "<UNRST>",
            "--init",
            "<INIT>",
            "--no_logging",
            "<NO_LOGGING>",
            "--debug",
            "<DEBUG>",
            "--residual_trapping",
            "<RESIDUAL_TRAPPING>",
            "--readable_output",
            "<READABLE_OUTPUT>",
            "--config_file_inj_wells",
            "<CONFIG_PLUME_TRACKING>",
            "--gas_molar_mass",
            "<GAS_MOLAR_MASS>",
        ]

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation:
        return ForwardModelStepDocumentation(
            description=_DESC_CO2_CONTAINMENT,
            category=_CATEGORY,
        )


class Co2PlumeAreaStep(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="CO2_PLUME_AREA",
            command=[
                shutil.which("co2_plume_area"),
                "<INPUT>",
                "--output_csv",
                "<OUTPUT_CSV>",
                "--no_logging",
                "<NO_LOGGING>",
                "--debug",
                "<DEBUG>",
            ],
            default_mapping={
                "<OUTPUT_CSV>": "share/results/tables/plume_area.csv",
                "<NO_LOGGING>": -1,
                "<DEBUG>": -1,
            },
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation:
        return ForwardModelStepDocumentation(
            description=_DESC_CO2_PLUME_AREA,
            category=_CATEGORY,
        )


class Co2PlumeExtentStep(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="CO2_PLUME_EXTENT",
            command=[
                shutil.which("co2_plume_extent"),
                "<CASE>",
                "--config_file",
                "<CONFIG_PLUME_EXTENT>",
                "--inj_point",
                "<INJ_POINT>",
                "--calc_type",
                "<CALC_TYPE>",
                "--output_csv",
                "<OUTPUT_CSV>",
                "--threshold_gas",
                "<THRESHOLD_GAS>",
                "--threshold_dissolved",
                "<THRESHOLD_DISSOLVED>",
                "--column_name",
                "<COLUMN_NAME>",
                "--no_logging",
                "<NO_LOGGING>",
                "--debug",
                "<DEBUG>",
            ],
            default_mapping={
                "<CONFIG_PLUME_EXTENT>": "",
                "<INJ_POINT>": "",
                "<CALC_TYPE>": "plume_extent",
                "<OUTPUT_CSV>": "share/results/tables/plume_extent.csv",
                "<THRESHOLD_GAS>": 0.2,
                "<THRESHOLD_DISSOLVED>": 0.0005,
                "<COLUMN_NAME>": "",
                "<NO_LOGGING>": -1,
                "<DEBUG>": -1,
            },
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation:
        return ForwardModelStepDocumentation(
            description=_DESC_CO2_PLUME_EXTENT,
            category=_CATEGORY,
        )


class Co2CsvArrowConverterStep(ForwardModelStepPlugin):
    def __init__(self):
        # This FORWARD_MODEL is slightly different from the others, as it
        # does not provide individual keywords for every possible argument.
        # Instead, it uses a single <OPTIONS> keyword that can be used to
        # pass any number of command line options to the underlying script.
        #
        # One advantage of this is less maintenance, as the underlying
        # script may change its command line options without requiring
        # changes to this FORWARD_MODEL.
        #
        # Another advantage is that we don't need placeholder defaults
        # (e.g. "-1") for every possible command line option, combined
        # with a parser pre-processing step that replaces these with
        # the actual defaults.
        #
        # The downside is that the user must know the command line options
        # of the underlying script, and that the command line options
        # are not documented in the FORWARD_MODEL documentation.
        #
        # Another downside is that this FORWARD_MODEL appears different
        # from the others, which may be confusing to users.

        super().__init__(
            name="CO2_CSV_ARROW_CONVERTER",
            command=[
                shutil.which("co2_csv_arrow_converter"),
                "--root_dir",
                "<ROOT_DIR>",
                "<OPTIONS>",
            ],
            default_mapping={
                "<ROOT_DIR>": ".",
                "<OPTIONS>": "",
            },
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation:
        return ForwardModelStepDocumentation(
            description=_DESC_CO2_CSV_ARROW_CONVERTER,
            category=_CATEGORY,
            examples=_EXAMPLES_CO2_CSV_ARROW_CONVERTER,
        )


class Grid3dAggregateMapStep(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="GRID3D_AGGREGATE_MAP",
            command=[
                shutil.which("grid3d_aggregate_map"),
                "--config",
                "<CONFIG_AGGREGATE>",
                "--eclroot",
                "<ECLROOT>",
                "--mapfolder",
                "<MAPFOLDER>",
                "--plotfolder",
                "<PLOTFOLDER>",
                "--folderroot",
                "<FOLDERROOT>",
                "--no_logging",
                "<NO_LOGGING>",
                "--debug",
                "<DEBUG>",
            ],
            default_mapping={
                "<ECLROOT>": -1,
                "<MAPFOLDER>": -1,
                "<PLOTFOLDER>": -1,
                "<FOLDERROOT>": -1,
                "<NO_LOGGING>": -1,
                "<DEBUG>": -1,
            },
            stderr_file="GRID3D_AGGREGATE_MAP.stderr",
            stdout_file="GRID3D_AGGREGATE_MAP.stdout",
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation:
        return ForwardModelStepDocumentation(
            description=_DESC_GRID3D_AGGREGATE_MAP,
            category=_CATEGORY,
            examples=_EXAMPLES_GRID3D_AGGREGATE_MAP,
        )


class Grid3dCo2MassMapStep(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="GRID3D_CO2_MASS_MAP",
            command=[
                shutil.which("grid3d_co2_mass_map"),
                "--config",
                "<CONFIG_CO2_MASS_MAP>",
                "--eclroot",
                "<ECLROOT>",
                "--mapfolder",
                "<MAPFOLDER>",
                "--plotfolder",
                "<PLOTFOLDER>",
                "--gridfolder",
                "<GRIDFOLDER>",
                "--folderroot",
                "<FOLDERROOT>",
                "--no_logging",
                "<NO_LOGGING>",
                "--debug",
                "<DEBUG>",
            ],
            default_mapping={
                "<ECLROOT>": -1,
                "<MAPFOLDER>": -1,
                "<PLOTFOLDER>": -1,
                "<GRIDFOLDER>": -1,
                "<FOLDERROOT>": -1,
                "<NO_LOGGING>": -1,
                "<DEBUG>": -1,
            },
            stderr_file="GRID3D_CO2_MASS_MAP.stderr",
            stdout_file="GRID3D_CO2_MASS_MAP.stdout",
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation:
        return ForwardModelStepDocumentation(
            description=_DESC_GRID3D_CO2_MASS_MAP,
            category=_CATEGORY,
            examples=_EXAMPLES_GRID3D_CO2_MASS_MAP,
        )


class Grid3dMigrationTimeStep(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="GRID3D_MIGRATION_TIME",
            command=[
                shutil.which("grid3d_migration_time"),
                "--config",
                "<CONFIG_MIGTIME>",
                "--eclroot",
                "<ECLROOT>",
                "--mapfolder",
                "<MAPFOLDER>",
                "--plotfolder",
                "<PLOTFOLDER>",
                "--folderroot",
                "<FOLDERROOT>",
                "--no_logging",
                "<NO_LOGGING>",
                "--debug",
                "<DEBUG>",
            ],
            default_mapping={
                "<ECLROOT>": -1,
                "<MAPFOLDER>": -1,
                "<PLOTFOLDER>": -1,
                "<FOLDERROOT>": -1,
                "<NO_LOGGING>": -1,
                "<DEBUG>": -1,
            },
            stderr_file="GRID3D_MIGRATION_TIME.stderr",
            stdout_file="GRID3D_MIGRATION_TIME.stdout",
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation:
        return ForwardModelStepDocumentation(
            description=_DESC_GRID3D_MIGRATION_TIME,
            category=_CATEGORY,
            examples=_EXAMPLES_GRID3D_MIGRATION_TIME,
        )


_DESC_CO2_CONTAINMENT = """
Calculates the amount of CO2 inside and outside a given perimeter, and
separates the result per formation and phase (gas/dissolved). Output is a table
on CSV format.

The most common use of the script is to calculate CO2 mass. Options for
calculation type input:

"mass": CO2 mass (kg), the default option
"cell_volume": CO2 volume (m3), a simple calculation finding the grid cells
with some CO2 and summing the volume of those cells
"actual_volume": CO2 volume (m3), an attempt to calculate a more precise
representative volume of CO2
"""


_DESC_CO2_PLUME_AREA = """
Calculates the area of the CO2 plume for each formation and time step, for both
SGAS and AMFG (Pflotran) / YMF2 (Eclipse).

Output is a table on CSV format.
"""


_DESC_CO2_PLUME_EXTENT = """
Calculates the maximum lateral distance of the CO2 plume from a given location,
for instance an injection point. It is also possible to instead calculate the
distance to a point or a line (north-south or east-west). The distances are
calculated for each time step, for both SGAS and AMFG (Pflotran) / XMF2
(Eclipse).

Output is a table on CSV format. Multiple calculations specified in the
YAML-file will be combined to a single CSV-file with many columns.
"""


_DESC_CO2_CSV_ARROW_CONVERTER = """
This scripts checks all FMU realizations for missing arrow or
csv files representing plume extent, area or containment data. If one exists,
but not the other, it will create the missing file.
"""


_DESC_GRID3D_AGGREGATE_MAP = "Aggregate property maps from 3D grids."


_DESC_GRID3D_CO2_MASS_MAP = """
Produces maps of CO2 mass per date, formation and phase (gas/dissolved).
Outputs are .gri files (one per requested combination of date, phase,
formation).

A yaml config file is the input file to co2_mass_maps. Through this file
the user can decide on which dates, phases or formations the maps are
produced. See tests/yaml for examples of yaml files.
"""


_DESC_GRID3D_MIGRATION_TIME = "Generate migration time property maps."


_EXAMPLES_CO2_CSV_ARROW_CONVERTER = """
If running from the root directory of the project, the default parameters
is probably what you want:

.. code-block:: console

  FORWARD_MODEL CO2_CSV_ARROW_CONVERTER()

The root directory can be specified with the <ROOT_DIR> parameter, and more
advanced options can be specified with the <OPTIONS> parameter. For example:

.. code-block:: console

  FORWARD_MODEL CO2_CSV_ARROW_CONVERTER(<ROOT_DIR>=/path/to/root, <OPTIONS>="--force_arrow_overwrite --realization_pattern=realization-*/iter-* --kept_columns=zone,plume_group")
"""  # noqa: E501


_EXAMPLES_GRID3D_AGGREGATE_MAP = """
.. code-block:: console

  FORWARD_MODEL GRID3D_AGGREGATE_MAP(<CONFIG_AGGREGATE>=conf.yml, <ECLROOT>=<ECLBASE>)
"""


_EXAMPLES_GRID3D_CO2_MASS_MAP = """
.. code-block:: console

  FORWARD_MODEL GRID3D_CO2_MASS_MAP(<CONFIG_CO2_MASS_MAP>=conf.yml, <ECLROOT>=<ECLBASE>)
"""


_EXAMPLES_GRID3D_MIGRATION_TIME = """
.. code-block:: console

  FORWARD_MODEL GRID3D_MIGRATION_TIME(<CONFIG_MIGTIME>=conf.yml, <ECLROOT>=<ECLBASE>)
"""


@ert_plugin(name="CCS_SCRIPTS")
def installable_forward_model_steps() -> list[ForwardModelStepPlugin]:
    return [
        Co2ContainmentStep,
        Co2PlumeAreaStep,
        Co2PlumeExtentStep,
        Co2CsvArrowConverterStep,
        Grid3dAggregateMapStep,
        Grid3dCo2MassMapStep,
        Grid3dMigrationTimeStep,
    ]
