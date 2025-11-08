# Framework-for-Passive-Rectifier-Topology-Selection-in-LLC-Resonant-Converters

The code in this repository refers to the paper "Decision Support Framework for Passive Rectifier Topology Selection in LLC Resonant Converters" 
presented in the 17th Seminar on Power Electronics and Control (SEPOC). IEEExplore link:

## How this repository is structured

- Folders "llc_sim" and "llc_simulator.egg-info"

  - Contains all the code needed for the electrical simulations (for a comprehensive look at this, refer to the paper "A Python Framework for Accelerated Design and Automated Analysis of LLC Resonant Converters" also presented in the 17th SEPOC. IEEExplore link: 

- Folder "SEPOC2025"

  - Contais all the code actually related to the paper, where:
    -  "magnetics_and_rectifier.py" contais the code related to the design routines and loss calculations;
    -  "diode_database.json" is the diodes database;
    -  "cores_shapes_params.ndjson" is the ferrite cores database;
    -  Subfolder "Diode_Curve_Fitting" contais the code for curve-fitting any diode VxI curves and obtain the 8 coefficients discussed in the paper.


- Files "CompleteExample - Case Study 01.ipynb" and "CompleteExample - Case Study 02.ipynb"
  - Are jupyter notebooks with the two case studies presented in the paper.

## How to use this repository

### Adding new cores to "cores_shapes_params.ndjson"

------------------------

### Adding new diodes to "diode_database.json"

------------------------

### Using the LLC electrical simulator

------------------------

### Using the Decision Support framework

#### transformer_design_params

------------------------


















