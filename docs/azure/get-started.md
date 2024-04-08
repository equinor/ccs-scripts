# Running on Azure

## 🎯 Goal

In order to run your simulations on Azure, you will need to have access to an Azure subscription and create a dedicated project space on the subscription. 
A step-by-step guide is available below.


## 📝 Set-by-step

1. Access to the following is a <span style="background-color: #DFF5FF">pre-requesite</span>:

    - [Unix Basic Access](https://accessit.equinor.com/Search/Search?term=UNIX+BASIC+ACCESS)
    - [HP Remote Graphics Receiver Windows](https://accessit.equinor.com/Search/Search?term=HP+REMOTE+GRAPHICS+RECEIVER+WINDOWS) 
<br />
<br />

2. To gain <span style="background-color: #DFF5FF">access to Azure</span> and <span style="background-color: #DFF5FF">create your project area</span>, send a request to Hans Rune Bue (TDI OG SUB HPC-team) with the following information. Content highlighted in bold needs to be changed according to your needs:

    - Email title:
    >"Request for extra CCS project into the S268 subscription"
    - Email core:
        - Create your project area and space: 
        >We request: 10 TB for /project/**[project_name]** and at least 10 TB for /scratch/**[project_name]** on subscription S268. 
        - Add your "on-premise" unix group if you have one and add short names of people who should have access:
        >The unix group owner should be **[unix_group_name]**, ref the **[unix_group_name]** group “On-Premise”: 
        >getent group: **[list of short names]**
        - Number of RGS machines:
        >We also request extra **[nb]** RGS-machines in the S268 RGS-pool if this is possible to deliver.

    You should now have access to Azure and have a dedicated area to work on - congratulations! 🏆
<br />
<br />
3. You will need to <span style="background-color: #DFF5FF">recreate your .cshrc file</span> in your home directory. This file contains all the system-wide settings (alias, visual settings, etc). 

## 📚 More resources

### Tips & tricks

- ResInsight: `/prog/ResInsight/script/resinsight`
<br />
<br />
- To check if jobs are running correctly:​

    `qstat` - Check if job is running (H: Hold, Q: Queue, R: Run)

    `qstat –f [jobID]` - Get more info on the run ​

    `qdel jobID` - Delete job
<br />
<br />

- How to synchronize folders between RGS nodes and Azure:

    Remember to change the highlighted sections.

    If you are using another subscirption than S268, update with your own subscription number.

| Command line      | Description |
| ----------- | ----------- |
| Rsync –anv <span style="background-color: #DFF5FF">22.1.2/</span> s268-<span style="background-color: #FFE7D6">rgs0001</span>.s268.oc.equinor.com:<span style="background-color: #DFF5FF">/project/fmu_for_ccs/22.1.2/</span>​ |Synchronize 22.1.2/ folder to Azure – dry run​ |
| Rsync –av <span style="background-color: #DFF5FF">22.1.2/</span> s268-<span style="background-color: #FFE7D6">rgs0001</span>.s268.oc.equinor.com:<span style="background-color: #DFF5FF">/project/fmu_for_ccs/22.1.2/</span>​ |Synchronize 22.1.2/ folder to Azure​​ |
| Rsync –av s268-<span style="background-color: #FFE7D6">rgs0001</span>.s268.oc.equinor.com:<span style="background-color: #DFF5FF">22.1.2/</span> <span style="background-color: #DFF5FF">/project/fmu_for_ccs/22.1.2/</span>​ |Synchronize folder from Azure to RGS​​ |


### Additional information & point of contact
For more information, please refer to the [presentation](https://statoilsrm.sharepoint.com/:p:/r/sites/SubOps/_layouts/15/Doc.aspx?sourcedoc=%7B122A0344-9D32-4B3C-BA6B-175B86B4F620%7D&file=Azure-S268-FirstUse.pptx&action=edit&mobileredirect=true) from Mark Reed and Hans Rune Bue. It dives into the configuration specifics of the Azure Europe West cluster in detail.

Contact people:

- Hans Rune Bue ([hrbu](emailto:hrbu@equinor.com)) - Azure
- Ketil Nummedal ([kenu](emailto:kenu@equinor.com)) - RGS
- Roger Nybø ([rnyb](emailto:rnyu@equinor.com)) - ERT 


