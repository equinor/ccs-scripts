# Overview

## About Weviz

Webviz is designed for analysing FMU model results.

More documentation on Webviz: [Drogon example](https://webviz-subsurface-example.azurewebsites.net/how-was-this-made)


## Available scripts

**Maps**

| Maps      | Description |
| ----------- | ----------- |
| [Migration time map](../webviz/maps/mig-time.md)     | Returns one map per formation showing the time it takes for the CO2 to migrate to a certain point (only SGAS).|
| [Aggregation map](../webviz/maps/agg-map.md)   | Returns one map per formation returning the highest saturation per location (in SGAS & AMFG). |
| [Mass map](../webviz/maps/mass-map.md)   | Returns one map per formation showing the aggregated mass of CO2.         |

**General scripts**

| Scripts      | Description |
| ----------- | ----------- |
| [Plume extent](../webviz/scripts/plume-extent.md)      | Calculates the maximum distance of the CO2 plume to the injector or another point (as gas or CO2 dissolved in water).|
| [Plume area](../webviz/scripts/plume-area.md)   | Calculates the plume area per formation in terms of CO2 as gas or CO2 dissolved in water (input: Max saturation maps). |
| [CO2 containment](../webviz/scripts/co2-containment.md)   | Calculates how much CO2 is inside or outside a boundary (as a volume or mass). Pre-requisite to the CO2 leakage plugin.|

**Webviz plugin**

| Plugin      | Description |
| ----------- | ----------- |
| [CO2 leakage](../webviz/plugin/co2-leakage.md)   | Plugin dedicated to surveiling and quantifying how much CO2 leaks outside the field's license boundary or any other area defined by the user.|


##ERT config file - example

Each of these scripts and command lines are detailed in there respected sections (general scripts, maps, plugin). 
Here is the post-processing steps in ERT config file with all available scripts. This step is a pre-requisite to ensemble analysis using Webviz:

~~~ yaml title="Example of post-processing steps in ERT config file"
{% include "./ert-config.yml" %}
~~~

