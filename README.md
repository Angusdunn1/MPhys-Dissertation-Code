# MPhys-Dissertation-Code
This repository contains all post-processing code used in my dissertation project on radiation shielding analysis using FLUKA. The goal of the project was to simulate how various materials attenuate proton radiation, particularly in the context of Solar Particle Events (SPEs) relevant to crewed space missions.

The FLUKA simulations were run using FLAIR, and this repository includes Python scripts for processing the resulting output data, calculating dose equivalents, and generating plots.

## Contents
### Data folder
Includes example input files, ASCII output, and a few FLUKA .inp samples for reference.

### Results folder
Includes the plots and a table for each material, displaying the thickness, dose per proton recieved by the phantom and the errors.

### Example_Dose_Sum.py
* Reads ASCII output from FLUKA (converted from .bnn), sums dose values from each bin, and normalises per primary proton. Outputs total dose equivalent in pSv/primary.
* Generates plots of dose equivalent vs. shielding thickness for all tested materials.

### absorption_coefficient.py
Performs curve fitting on dose-thickness data to estimate absorption coefficients.