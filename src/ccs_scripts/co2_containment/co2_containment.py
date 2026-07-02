#!/usr/bin/env python
"""
Calculates the amount of CO2 inside and outside a given perimeter,
and separates the result per formation and phase (gas/dissolved).
Output is a table in CSV format.
"""

from ccs_scripts.co2_containment.co2_calculation import calculate_co2
from ccs_scripts.co2_containment.containment_calculation import calculate_containment
from ccs_scripts.co2_containment.input import init_timer, process_input
from ccs_scripts.co2_containment.output import export_results
from ccs_scripts.co2_containment.source_data import extract_source_data
from ccs_scripts.utils.timer import Timer


def main() -> None:
    """
    Takes input arguments and calculates total co2 mass or volume at each time
    step, divided into different phases and locations. Creates a data frame,
    then exports the data frame to a csv file.
    """
    init_timer()
    timer = Timer()
    timer.start("total")

    (
        args,
        zone_info,
        region_info,
        calc_type,
        cont_polygon,
        nogo_polygon,
        plume_groups,
    ) = process_input()

    source_data, _ = extract_source_data(
        args.egrid,
        args.unrst,
        zone_info,
        region_info,
        args.residual_trapping,
        args.init,
    )
    co2_data = calculate_co2(
        source_data,
        calc_type,
        args.residual_trapping,
        args.cirrus_info_file,
    )
    containment_data = calculate_containment(
        co2_data,
        cont_polygon,
        nogo_polygon,
        calc_type,
        zone_info.int_to_zone,
        region_info.int_to_region,
        args.residual_trapping,
        plume_groups,
    )
    export_results(
        containment_data,
        args.calc_type_input,
        args.out_dir,
        zone_info.int_to_zone,
        region_info.int_to_region,
        args.residual_trapping,
        args.readable_output,
    )

    timer.stop("total")
    timer.report()


if __name__ == "__main__":
    main()
