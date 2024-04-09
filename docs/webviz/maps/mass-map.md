# Mass maps

## 🎯 Overview

![image alt ><](./img/mass-map.jpg)

## 📝 How to set it up?

### ERT

❌ Not available on Komodo yet! A local installation is needed for now. Follow the instructions below to test it:

 1. <span style="background-color: #DFF5FF">Clone the repository</span> locally:
   ```yaml
   git clone https://github.com/AudunSektnanNR/xtgeoapp-grd3dmaps-as.git --branch develop
   ```
<br />
<br />

 2. Create a <span style="background-color: #DFF5FF">local environment</span>. The example below creates a local env based on Komodo Bleeding, called "kenv-b". Note that this line is valid when using Azure, not on-prem sessions.
   ```yaml
   komodoenv -r /prog/komodo/bleeding-py38 kenv-b --root /prog/komodo/
   ```
<br />
<br />

 3. <span style="background-color: #DFF5FF">Source</span> your local environment:
   ```yaml
   source kenv-b/enable.csh
   ```
<br />
<br />   

 4. <span style="background-color: #DFF5FF">Install the repository</span> on your local environment:
   ```yaml
   pip install xtgeoapp-grd3dmaps-as/
   ```
<br />
<br />

 5. Add the following <span style="background-color: #DFF5FF">config file</span> in `ert/bin/jobs`:
    ```yaml
    STDERR    co2_mass.stderr
    STDOUT    co2_mass.stdout

    EXECUTABLE    [Path to grid3d_co2_mass.py from this directory] 


    ARGLIST        <XARG1> <XARG2> <XARG3> <XARG4>
    ```
<br />
<br />

 6. <span style="background-color: #DFF5FF">Install the job locally</span> in `ert/input/config/install_custom_jobs.ert`:
    ```yaml
    INSTALL_JOB CO2_MASS_MAP     ../../bin/jobs/CO2_MASS_MAP
    ```
<br />
<br />

 7. Add the <span style="background-color: #DFF5FF">config file</span> named `grid3d_co2_mass_map.yml` under `ert/input/config`. See config file example in the section below.
 <br />
 <br />

 8. <span style="background-color: #DFF5FF">Add the forward model</span> in your ert-config, located under `ert/input/model`:
    ```yaml
    FORWARD_MODEL CO2_MASS_MAP(<XARG1>="-c",<XARG2>=<CONFIG>/../input/config/grid3d_co2_mass_map.yml, <XARG3>="--eclroot", <XARG4>=<ECLBASE>)
    ```


### Config file

Config file are to be placed in `ert/input/config` folder. Here is a template example:

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