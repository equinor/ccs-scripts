# ccs-scripts

:scroll: **ccs-scripts** gathers a collection of post-processing scripts dedicated to CCS outputs from Eclipse and Pflotran.

>Note: These scripts are beeing tested and frequently updated, new releases will be available occasionally :recycle:


---
## Functionalities

### Available in the repositoy

- **plume_extent:** Calculates the maximum extent of the plume from a selected injector well. 

- **plume_area:** Calculates the plume area for CO2 as free gas and dissolved in water.

- **co2_containment:** This scripts can output several different information: the plume mass, plume volume and returns volumes of CO2 inside/outside a boundary when 1 or 2 polygons are provided. 

### Additional functionalities
Additional fonctionnalities are available to post-process CO2 storage modeling data. They have also been developed by the Dig CCS Sub team but are located in other repositories. More information below: 

- **maps:** 3 maps are available

Migration time map: This script outputs a map where the plume is displayed in terms of time it takes to travel from the injection point. 

Aggregate map: Returns a map of maximum aggregation through all layers ("worse" case scenario).

Mass map: Displays mass of CO2 as a map. Mass maps can be exported per formation and time step. 

Documentation: [link](https://fmu-docs.equinor.com/docs/ert/reference/configuration/forward_model.html#GRID3D_MIGRATION_TIME)

- **plugin:** The CO2 leakage plugin can be used on Webviz to visualize the CO2 plume, quantities inside / outside a boundary / region, etc. 

Documentation: [link](https://equinor.github.io/webviz-subsurface/#/webviz-subsurface?id=co2leakage)



## Installation 

This repository is currently beeing linked to Komodo and ERT. In the meantime, ccs-scripts can be cloned and installed on your local komodo environment using pip install:

```sh
pip install ccs-scripts
```

## Developing & Contributing

Do you have a script that processes CCS data? Would like to share it with the CCS community? **Contributions are welcome!** :star_struck: Please take contact with Floriane Mortier (fmmo).

## Documentation

A brand new documentation is now available on the following site: [link](https://fmu-for-ccs.radix.equinor.com). 

It gathers definitions, tutorials, theory behind the calculations, etc. 
