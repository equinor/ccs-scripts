# ccs-scripts

:scroll: **ccs-scripts** gathers a collection of post-processing scripts dedicated to CCS outputs from Eclipse and Pflotran.

>Note: These scripts are beeing tested and frequently updated, new releases will be available occasionally :recycle:


---
## Functionalities

### Available in the repositoy

- **co2_plume_extent:** Calculates the maximum extent of the plume from a selected injector well. 

- **co2_plume_area:** Calculates the plume area for CO2 as free gas and dissolved in water.

- **co2_plume_tracking:** Tracks and differenciates the different CO2 plumes from each other. Input information to other scripts.

- **co2_containment:** This scripts can output several different information: the plume mass, plume volume and returns volumes of CO2 inside/outside a boundary when 1 or 2 polygons are provided. 

- **grid3d_migration_time:** Returns one map per formation showing the time it takes for the CO2 to migrate to a certain point (only SGAS ).

- **grid3d_aggregate_map:** Returns one map per formation returning the highest saturation per location (in SGAS & AMFG).

- **grid3d_co2_mass_map:** Returns one map per formation showing the aggregated mass of CO2 (Free, Dissolved & Total).

### Result visualization
Webviz can be used to visualize the results generated by the scripts. A specific plugin was created for CCS purposes: 

The **CO2 leakage plugin** can be used on Webviz to visualize the CO2 plume, quantities inside / outside a boundary / region, etc. 

Documentation: [link](https://equinor.github.io/webviz-subsurface/#/webviz-subsurface?id=co2leakage)



## Installation 

This repository is currently beeing linked to Komodo and ERT. In the meantime, ccs-scripts can be cloned and installed on your local komodo environment using pip install:

```sh
pip install ccs-scripts
```

## Developing & Contributing

Do you have a script that processes CCS data? Would like to share it with the CCS community? **Contributions are welcome!** :star_struck: Please take contact with Floriane Mortier (fmmo).

See [contribution](https://github.com/equinor/ccs-scripts/blob/main/CONTRIBUTING.md) for details.

## Documentation

A brand new documentation is now available on the following site: [link](https://fmu-for-ccs.radix.equinor.com). 

It gathers definitions, tutorials, theory behind the calculations, etc. 
