
CO2_MASS_MAPS
===============

.. argparse::
   :module: ccs_scripts.co2_mass_maps.co2_mass_maps
   :func: get_parser
   :prog: co2_mass_maps

Produces maps of CO\ :sub:`2` mass per date, fomation and phase (gas/dissolved). Outputs are .gri files (one per requested combination of date, phase, formation).

A yaml config file is the input file to co2_mass_maps. Through this file the user can decide for which dates, phases or formations the maps are produced. See tests/yaml for examples of yaml files. (WIP)

(Ellaborate more on the names of the maps produced)

