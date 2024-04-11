# CO2 containment

## 🎯 Overview

Calculates the amount of CO2 inside or outside a boundary (polygons, zones or regions) and returns it as a volume or mass. Multiple options are available.


## 📝 How to set it up?

### ERT

✅ Available on Komodo

⚪ Optional parameters: Boundary polygons / zone output file / regions - depending on which calculation method has been selected. For more information on how to output your boundary polygons or zone output file, read the "RMS" section below.

``` yaml title="Mandatory arguments"
<CASE>: Path to Eclipse case, including base name, but excluding the file extension (.EGRID, .INIT, .UNRST)

<CALC_TYPE_INPUT>: mass / cell_volume / actual_volume.
```
``` yaml title="Optional arguments"
<ROOT_DIR>: Path to root directory.

<OUT_DIR>: Path to output directory. By default, the filename is set to "plume_<calculation_type>.csv" and stored in <root_dir>/share/results/tables.

<CONTAINMENT_POLYGON>: Path to the containment polygon. Counts all CO2 as contains if not provided. 

<HAZARDOUS_POLYGON>: Path to the hazardous polygon.

<EGRID>: Path to EGRID file. Overwrites <case> if provided.

<UNRST>:  Path to UNRST file. Overwrites <case> if provided.

<INIT>:  Path to INIT file. Overwrites <case> if provided.

<ZONEFILE>: Path to file containing zone information. By default, does not calculates Co2 containment per zone.
```

Example:

``` yaml title="Calculates actual volume of CO2 in the model"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <CALC_TYPE_INPUT>="actual_volume")
```


### RMS

RMS outputs some key information that the co2_containment script needs in order to function. Necesary jobs and output examples are provided below:

**Boundary polygons**

1. Create your polygons in the clipboard, under a `boundary_polygon` folder. 

2. To export your boundary via RMS, use the following python job:
``` yaml title="export_boundary_polygons.py"
###############################################################################
#export boundary polygons(csv-file) for use with webviz co2leakage plugin.
#
#Polygons have to be manually created by the user in a clipboard folder.
#Folder name and boundary name needs to be defined by the user.
#
#modified aug 2022 by sbern based on export_fault_polygons.py script made by rnyb (aug 2020) 
###############################################################################

import xtgeo
from pathlib import Path

# ------------------
# user input
# ------------------

# define polygon to export (in clipboard folder)
BOUNDARY_FOLDER = "boundary_polygon"
BOUNDARY_NAME = ["containment","hazardous"]

# ------------------
# functions
# ------------------
def _xtract_boundary_polygon(boundary_name, boundary_folder):
    mypoly = xtgeo.polygons_from_roxar(project, boundary_name, boundary_folder, 'clipboard')
    return mypoly.dataframe, mypoly


def export_boundary_polygon(boundary_name, boundary_folder):
    path = Path("../../share/results/polygons")
    csvname = ""+boundary_name+"--boundary.csv"

    path.mkdir(parents=True, exist_ok=True)
    csvfile = path / csvname

    # get polygon
    df_pol, pol = _xtract_boundary_polygon(boundary_name, boundary_folder)
    # export polygon
    df_pol.to_csv(csvfile, index=False)
    print("Created file {name}".format(name=csvfile))


# ------------------
# main
# ------------------

if __name__ == "__main__":
    for boundname in BOUNDARY_NAME:
        export_boundary_polygon(boundname, BOUNDARY_FOLDER)
    print("Done")

```

3. The output file should look like:
``` yaml title="Example of a boundary polygon exported by RMS"
X_UTME,Y_UTMN,Z_TVDSS,POLY_ID
565261.4599609375,6711611.201171875,1730.358154296875,0
557894.8095703125,6705717.87890625,1901.104248046875,0
564719.4873046875,6698037.1328125,1794.671630859375,0
572294.3525390625,6704132.8828125,1602.563232421875,0
565243.28515625,6711619.79296875,1730.311279296875,0
565261.4599609375,6711611.201171875,1730.358154296875,0
```

**Zone output file**

The following RMS job is used to export the zone table:

``` yaml title="make_zones_layer_table.py"
##############################################################################
#
# NAME:
#    make_zones_layer_table.py
#
# AUTHOR(S):
#    Jimmy Zurcher (jiz@equinor.com)
#
# DESCRIPTION:
#    Create a text table with the zone associated with each layer of a 3D grid
#
# rnyb, Sep 20 - export of zone-layer info as part of the main workflow to ensure 
#                consistency when running ert (gendata_rft) 
#                (ert gendata_rft keyword should be set up to point to this file)
###############################################################################

grid_name = 'Main_grid'
file_name = 'layer_zone_table'
file_path = '../../rms/output/zone/'

grid = project.grid_models[grid_name].get_grid()
indexer = grid.grid_indexer
dimensions = indexer.dimensions

n_layers = dimensions[2]

txt_file = open(file_path+file_name+'.txt', 'w')
csv_file = open(file_path+file_name+'.csv', 'w')
csv_file.write('LAYER,ZONE\n')

print('Layer   Zone')
for k in range(0, n_layers):
    zone_name = 'FAIL'
    for zone_index in indexer.zonation:
        for layer_interval in indexer.zonation[zone_index]:
        # More than one group of layers only if repeat sections in the grid
            if k in layer_interval:
                zone_name = grid.zone_names[zone_index]
    if k < 9:
        sep = '    '
    else:
        sep = '   '
    print('{}{}{}'.format(k+1, sep, zone_name))
    txt_file.write('{}{}{}\n'.format(k+1, sep, zone_name))
    csv_file.write('{},{}\n'.format(k+1, zone_name))

txt_file.close()
csv_file.close()

print('Done.')
```

The output file should look like:

``` yaml title="Example of `layer_zone_file.csv` exported by RMS"
LAYER,ZONE
1,Sognefjord
2,Sognefjord
3,Sognefjord
4,Sognefjord
5,Sognefjord
[...]
310,UpperLunde
311,LowerLunde
```

## 📚 Other examples

Note: rename `[1st_oundary_name]` and `[2nd_boundary_name]` according to your project's naming standards.

``` yaml title="Calculates cell volume with CO2 in the model"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <CALC_TYPE_INPUT>="cell_volume")
```
``` yaml title="Calculates mass of CO2 in the model"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <CALC_TYPE_INPUT>="mass")
```

``` yaml title="Calculates actual volume of CO2 inside & outside 2 polygons"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <OUT_DIR>="share/results/tables/plume_actual_volume.csv", <CALC_TYPE_INPUT>="actual_volume", <CONTAINMENT_POLYGON>=share/results/polygons/[1st_boundary_name]--boundary.csv, <HAZARDOUS_POLYGON>=share/results/polygons/[2nd_boundary_name]--boundary.csv, <ZONEFILE>=<RUNPATH>rms/output/zone/layer_zone_table.csv)
```


``` yaml title="Calculates cell volume inside & outside your 1 polygon"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <OUT_DIR>="share/results/tables/plume_cell_volume.csv", <CALC_TYPE_INPUT>="cell_volume", <CONTAINMENT_POLYGON>=share/results/polygons/[1st_boundary_name]--boundary.csv)
```

``` yaml title="Calculates mass of CO2 inside & outside your 2 polygons and per zone"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <OUT_DIR>="share/results/tables/plume_mass.csv", <CALC_TYPE_INPUT>="mass", <CONTAINMENT_POLYGON>="share/results/polygons/[1st_boundary_name]--boundary.csv",  <HAZARDOUS_POLYGON>="share/results/polygons/[2nd_boundary_name]--boundary.csv", <ZONEFILE>="<RUNPATH>rms/output/zone/layer_zone_table.csv" )
```


## 🔧 Versions & Updates

**Future development**

In progress:

- Calculates CO2 containment per regions

- Added a "--verbose" option. Outputs all the calculation steps during the ERT run in the .STDERR file.

``` yaml title="Calculates actual volume using region property for region calculation"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>,  <CALC_TYPE_INPUT>="actual_volume", <CONTAINMENT_POLYGON>="share/results/polygons/[1st_boundary_name]--boundary.csv", <HAZARDOUS_POLYGON>="share/results/polygons/[2nd_boundary_name]--boundary.csv", <REGION_PROPERTY>="FIPSEG", <ZONEFILE>="rms/output/zone/zonation_geo_map.yml", <XARG1>="--verbose")
```
<br />
<br />

**Updates**

- This script know returns volume and mass of CO2 inside and outside zones in addition to polygons. 
