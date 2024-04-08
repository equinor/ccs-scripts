# Mass maps

## 🎯 Overview

![image alt ><](mass-map.jpg)

## 📝 How to set it up?

### ERT

❌ Not available on Komodo yet! A local installation is needed for now. Follow the instructions below to test it:

 1. Clone the following repository: `https://github.com/AudunSektnanNR/xtgeoapp-grd3dmaps-as/tree/develop`
<br />
<br />

 2. Install it on your local environment.
<br />
<br />

 3. Add the following file in `ert/bin/jobs`:
    ```yaml
    STDERR    co2_mass.stderr
    STDOUT    co2_mass.stdout

    EXECUTABLE    [Path to grid3d_co2_mass.py from this directory] 


    ARGLIST        <XARG1> <XARG2> <XARG3> <XARG4>
    ```

 4. Add the following line in `ert/input/config/install_custom_jobs.ert`:
    ```yaml
    INSTALL_JOB CO2_MASS_MAP     ../../bin/jobs/CO2_MASS_MAP
    ```

 5. Add the `grid3d_co2_mass_map.yml` file under `ert/input/config`. See config file example in the section below.
 <br />
 <br />

 6. Add the following line in your ert-config, located under `ert/input/model`:
    ```yaml
    FORWARD_MODEL CO2_MASS_MAP(<XARG1>="-c",<XARG2>=<CONFIG>/../input/config/grid3d_co2_mass_map.yml, <XARG3>="--eclroot", <XARG4>=<ECLBASE>)
    ```


### Config file

Config file are to be placed in `ert/input/config`

Template example:

~~~ yaml title="grid3d_co2_mass_map.yml"
{% include "./config-file-examples/mass-template2.yml" %}
~~~

## 📚 Other config file example

*More to come*

## 🔧 Versions & updates

**Future development**

- Bring this solution into komodo - Work in progress
<br />
<br />

**Updates**