#Tutorials & Resources

## Pflotran

- [Documentation](https://docs.opengosim.com/)
- [Tutorials](https://docs.opengosim.com/tutorial/tutorials/)


## Webviz

**General**

Webviz is designed for analysing FMU model results.

- [Introduction to Webviz](https://equinor.github.io/webviz-subsurface/#/)
- [Webviz config file](https://webviz-subsurface-example.azurewebsites.net/how-was-this-made-yaml-config-file)
- [Webviz subsurface plugins](https://equinor.github.io/webviz-subsurface/#/webviz-subsurface)


**CO2 storage**

CO2 storage synthetic model with Webviz visualization and config file examples: *Links to come*


## Local tests & cloning repositories

Here is a guide on how to clone the ccs-scripts repository and install it locally. By doing so, you'll be able to test all our latest updates and modify the scripts yourself! Developed an extension to one of the scripts? Have a great development idea? Let us know so we can add your contribution!


 1. <span style="background-color: #DFF5FF">Clone the repository</span> locally. Might be easier to put it in the `ert/bin/script` folder, but depends on personal preferences:
   ```yaml title="Main branch"
   git clone https://github.com/equinor/ccs-scripts.git
   ```
   Or
   ```yaml title="Develop branch"
   git clone https://github.com/equinor/ccs-scripts.git --branch develop
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
   pip install ccs-script/
   ```
<br />
<br />

 5. If you want to <span style="background-color: #DFF5FF">test a specific script</span>, remember to update the `EXECUTABLE` section in the `ert/bin/jobs` of the file. You can do this by entering the path to the new script. 