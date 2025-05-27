import getpass
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime

from ccs_scripts.aggregate._config import RootConfig


def log_input_configuration(config_: RootConfig, calc_type: str = "aggregate") -> None:
    """
    Log the provided input
    """
    version = "v0.9.2"
    is_dev_version = True
    if is_dev_version:
        version += "_dev"
        try:
            source_dir = os.path.dirname(os.path.abspath(__file__))
            short_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], cwd=source_dir
                )
                .decode("ascii")
                .strip()
            )
        except subprocess.CalledProcessError:
            short_hash = "-"
        version += " (latest git commit: " + short_hash + ")"

    col1 = 37
    now = datetime.now()
    date_time = now.strftime("%B %d, %Y %H:%M:%S")
    if calc_type == "aggregate":
        logging.info("CCS-scripts - Aggregate maps")
        logging.info("============================")
    elif calc_type == "time_migration":
        logging.info("CCS-scripts - Time migration maps")
        logging.info("=================================")
    elif calc_type == "co2_mass":
        logging.info("CCS-scripts - CO2 mass maps")
        logging.info("===========================")
    logging.info(f"{'Version':<{col1}} : {version}")
    logging.info(f"{'Date and time':<{col1}} : {date_time}")
    logging.info(f"{'User':<{col1}} : {getpass.getuser()}")
    logging.info(f"{'Host':<{col1}} : {socket.gethostname()}")
    logging.info(f"{'Platform':<{col1}} : {platform.system()} ({platform.release()})")
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    logging.info(f"{'Python version':<{col1}} : {py_version}")

    if calc_type == "co2_mass":
        logging.info(f"\n{'Unit':<{col1}} : tons")

    logging.info("\nInput configuration:")
    logging.info(f"{'  Grid file':<{col1}} : {config_.input.grid}")
    if not os.path.isabs(config_.input.grid):
        logging.info(
            f"{'    => Absolute path':<{col1}} : "
            f"{os.path.abspath(config_.input.grid)}"
        )
    if calc_type != "co2_mass":
        logging.info("  Properties:")
        if config_.input.properties is None:
            logging.info("    No properties specified")
        else:
            for p in config_.input.properties:
                logging.info(f"{'    - Name':<{col1}} : {p.name}")
                logging.info(
                    f"{'      Source':<{col1}} : "
                    f"{p.source if p.source is not None else '-'}"
                )
                logging.info(
                    f"{'      Lower threshold':<{col1}} : "
                    f"{p.lower_threshold if p.lower_threshold is not None else '-'}"
                )
    if len(config_.input.dates) > 0:
        logging.info(f"{'  Dates':<{col1}} : {', '.join(config_.input.dates)}")
    else:
        logging.info(f"{'  Dates':<{col1}} : - (not specified => using all dates)")

    op = config_.output
    logging.info("\nOutput configuration:")
    logging.info(f"{'  Map folder':<{col1}} : {op.mapfolder}")
    if not os.path.isabs(op.mapfolder):
        logging.info(
            f"{'    => Absolute path':<{col1}} : " f"{os.path.abspath(op.mapfolder)}"
        )
    if op.plotfolder is not None:
        logging.info(f"{'  Plot folder':<{col1}} : {op.plotfolder}")
        if not os.path.isabs(op.plotfolder):
            logging.info(
                f"{'    => Absolute path':<{col1}} : "
                f"{os.path.abspath(op.plotfolder)}"
            )
    else:
        logging.info(f"{'  Plot folder':<{col1}} : - (plot export not selected)")

    if calc_type == "co2_mass":
        if op.gridfolder is not None:
            logging.info(f"{'  Grid folder':<{col1}} : {op.gridfolder}")
            if not os.path.isabs(op.gridfolder):
                logging.info(
                    f"{'    => Absolute path':<{col1}} : "
                    f"{os.path.abspath(op.gridfolder)}"
                )
        else:
            logging.info(
                f"{'  Grid folder':<{col1}} : - "
                f"(not specified, so temp exported 3D grid files will be deleted)"
            )
    else:
        logging.info(f"{'  Grid folder':<{col1}} : - (only relevant for co2 mass maps)")
    logging.info(
        f"{'  Use lower case in file names':<{col1}} : {_bool_str(op.lowercase)}"
    )
    logging.info(
        f"{'  Module/method for 2D plots':<{col1}} : "
        f"{'plotly library' if op.use_plotly else 'quickplot from xtgeoviz'}"
    )
    logging.info(
        f"{'  Add tag to file name for aggr. maps':<{col1}} : "
        f"{_bool_str(op.aggregation_tag)}"
    )
    logging.info(
        f"{'  Replace masked values with zeros':<{col1}} : "
        f"{_bool_str(op.replace_masked_with_zero)}"
    )
    logging.info(f"{'  Mask zeros':<{col1}} : " f"{_bool_str(op.mask_zeros)}")

    logging.info("\nComputation configuration:")
    logging.info(
        f"{'  Aggregation method':<{col1}} : {config_.computesettings.aggregation.name}"
    )
    logging.info(
        f"{'  Weight by dz':<{col1}} : "
        f"{_bool_str(config_.computesettings.weight_by_dz)}"
    )
    logging.info(
        f"{'  Make maps for full grid (all zones)':<{col1}} : "
        f"{_bool_str(config_.computesettings.all)}"
    )
    logging.info(
        f"{'  Make maps per zone':<{col1}} : "
        f"{_bool_str(config_.computesettings.zone)}"
    )
    logging.info(
        f"{'  Calculate aggregate maps':<{col1}} : "
        f"{_bool_str(config_.computesettings.aggregate_map)}"
    )
    logging.info(
        f"{'  Calculate indicator maps':<{col1}} : "
        f"{_bool_str(config_.computesettings.indicator_map)}"
    )

    zon = config_.zonation
    logging.info("\nZonation configuration:")
    if not config_.computesettings.zone:
        logging.info(
            "(Note that these are not used since zone "
            "in computesettings is set to 'no')"
        )
    logging.info("  Z-property:")
    if zon.zproperty is None:
        logging.info("    No z-property specified")
    else:
        logging.info(f"{'    Source':<{col1}} : {zon.zproperty.source}")
        if not os.path.isabs(zon.zproperty.source):
            logging.info(
                f"{'      => Absolute path':<{col1}} : "
                f"{os.path.abspath(zon.zproperty.source)}"
            )
        logging.info(
            f"{'    Name':<{col1}} : "
            f"{zon.zproperty.name if zon.zproperty.name is not None else '-'}"
        )
        logging.info("    Zones:")
        zones = zon.zproperty.zones
        if len(zones) == 0:
            logging.info("      No zones specified")
        else:
            for z in zones:
                for i, (k, v) in enumerate(z.items()):
                    if i == 0:
                        logging.info(f"{f'      - {k}':<{col1}} : {v}")
                    else:
                        logging.info(f"{f'        {k}':<{col1}} : {v}")
    logging.info("  Z-ranges:")
    if len(zon.zranges) == 0:
        logging.info("    No z-ranges specified")
    else:
        for zr in zon.zranges:
            for i, (key, v2) in enumerate(zr.items()):
                if i == 0:
                    logging.info(f"{f'    - {key}':<{col1}} : {v2}")
                else:
                    logging.info(f"{f'      {key}':<{col1}} : {v2}")

    logging.info("\nMap configuration:")
    ms = config_.mapsettings
    if ms.templatefile is not None:
        logging.info("  Using template file (Option 1)")
    elif ms.xori is not None:
        logging.info("  Will use Option 2 since no template file has been specified")
    else:
        logging.info(
            "  Neither template file nor Origo x (etc) is specified,"
            " so will use pixel-to-cell-size ratio (Option 3)"
        )
    logging.info("  Option 1:")
    logging.info(
        f"{'    Template file':<{col1}} : "
        f"{ms.templatefile if ms.templatefile is not None else '- (not specified)'}"
    )
    if ms.templatefile is not None and not os.path.isabs(ms.templatefile):
        logging.info(
            f"{'      => Absolute path':<{col1}} : "
            f"{os.path.abspath(ms.templatefile)}"
        )
    logging.info("  Option 2:")
    logging.info(f"{'    Origo x':<{col1}} : {ms.xori if ms.xori is not None else '-'}")
    logging.info(f"{'    Origo y':<{col1}} : {ms.yori if ms.yori is not None else '-'}")
    logging.info(
        f"{'    Increment x':<{col1}} : {ms.xinc if ms.xinc is not None else '-'}"
    )
    logging.info(
        f"{'    Increment y':<{col1}} : {ms.yinc if ms.yinc is not None else '-'}"
    )
    logging.info(
        f"{'    Number of columns (x)':<{col1}} : "
        f"{ms.ncol if ms.ncol is not None else '-'}"
    )
    logging.info(
        f"{'    Number of rows (y)':<{col1}} : "
        f"{ms.nrow if ms.nrow is not None else '-'}"
    )
    if ms.xinc is not None and ms.ncol is not None:
        logging.info(f"{'    => Size x-direction':<{col1}} : {ms.xinc * ms.ncol}")
    if ms.yinc is not None and ms.nrow is not None:
        logging.info(f"{'    => Size y-direction':<{col1}} : {ms.yinc * ms.nrow}")
    logging.info("  Option 3:")
    logging.info(f"{'    Pixel-to-cell-size ratio':<{col1}} : {ms.pixel_to_cell_ratio}")

    cms = config_.co2_mass_settings
    if calc_type == "co2_mass" and cms is not None:
        logging.info("\nCO2 mass configuration:")
        logging.info(f"{'  UNRST source':<{col1}} : {cms.unrst_source}")
        if not os.path.isabs(cms.unrst_source):
            logging.info(
                f"{'    => Absolute path':<{col1}} : "
                f"{os.path.abspath(cms.unrst_source)}"
            )
        logging.info(f"{'  INIT source':<{col1}} : {cms.init_source}")
        if not os.path.isabs(cms.init_source):
            logging.info(
                f"{'    => Absolute path':<{col1}} : "
                f"{os.path.abspath(cms.init_source)}"
            )
        txt = "(not specified => calculating all maps)"
        logging.info(
            f"{'  Maps to calculate':<{col1}} : "
            f"{cms.maps if cms.maps is not None else f'- {txt}'}"
        )
        logging.info(
            f"{'  Include residual trapping':<{col1}} : "
            f"{_bool_str(cms.residual_trapping)}"
        )


def _bool_str(value: bool):
    return "yes" if value else "no"
