
CO2_PLUME_EXTENT
================

.. argparse::
   :module: ccs_scripts.co2_plume_extent.co2_plume_extent
   :func: get_parser
   :prog: plume_extent

Calculates the maximum lateral distance of the CO\ :sub:`2` plume from a given location,
for instance an injection point. It is also possible to instead calculate the distance
to a point or a line (north-south or east-west). The distances are calculated for each
time step, for both SGAS and AMFG (Pflotran) / YMF2 (Eclipse). It is possible to
use an YAML-file to set up multiple calculations.

Output is a table on CSV format. Multiple calculations specified in the YAML-file
will be combined to a single CSV-file with many columns.

CSV file example - plume extent
-------------------------------
Example of how the plume extent output CSV file is structured:

.. list-table:: CSV file of CO2 plume extent (m)
   :widths: 25 25 25 25 25
   :header-rows: 1

   * - DATE
     - MAX_DISTANCE_PlumeExtent_1_SGAS
     - MAX_DISTANCE_PlumeExtent_1_AMFG
     - MIN_DISTANCE_Point_1_SGAS
     - MIN_DISTANCE_Point_1_AMFG
   * - 2020-01-01
     - 0.0
     - 0.0
     - 3000.0
     - 3000.0
   * - 2060-01-01
     - 703.4
     - 761.5
     - 2612.0
     - 2691.0
   * - 2100-01-01
     - 1305.2
     - 1521.0
     - 2151.0
     - 2402.0
