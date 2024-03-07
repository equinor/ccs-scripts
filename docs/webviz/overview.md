# Overview

## About Weviz

Webviz is designed for analysing FMU model results.

More documentation on Webviz: [Drogon example](https://webviz-subsurface-example.azurewebsites.net/how-was-this-made)


## Available scripts

**Maps**

| Maps      | Description |
| ----------- | ----------- |
| Migration time map      | Returns one map per formation showing the time it takes for the CO2 to migrate to a certain point (only SGAS).|
| Aggregation map   | Returns one map per formation returning the highest saturation per location (in SGAS & AMFG). |
| Mass map   | Returns one map per formation showing the aggregated mass of CO2.         |

**General scripts**

| Scripts      | Description |
| ----------- | ----------- |
| Plume extent      | Calculates the maximum distance of the CO2 plume to the injector or another point (as gas or CO2 dissolved in water).|
| Plume area   | Calculates the plume area per formation in terms of CO2 as gas or CO2 dissolved in water (input: Max saturation maps). |
| CO2 containment   | Calculates how much CO2 is inside or outside a boundary (as a volume or mass). Pre-requisite to the CO2 leakage plugin.|

**Webviz plugin**

| Plugin      | Description |
| ----------- | ----------- |
| CO2 leakage   | Plugin dedicated to surveiling and quantifying how much CO2 leaks outside the field's license boundary or any other area defined by the user.|


##ERT config file - example

Screenshot of ERT config file with all available scripts.