# USRA-2020-AlexiMorin
Summer USRA project

Main codes and outputs produced during my summer at SFU glaciology group. 

Codes:
- compare_alpha.py
  Code used for testing alpha_GPR from Langhammer et. al. (2018) (see https://tc.copernicus.org/articles/13/2189/2019/)
  Plots produced are in the folder alpha_compare. More information in the report.
  
- compare_every_glaciers.py and comparing_models.py
  Code used at the beginning of the summer to compute the error from the test cases from ITMIX (see https://tc.copernicus.org/articles/11/949/2017/)
  Writing this code served as good practice for learning to manipulate georeferenced data and is probably bad. It also inspired the need and creation of the pyGM tool.
  
- dhdt.py
  Code used to compute metrics from the dh/dt data handed out by Étienne Berthier. Not much ended up coming out from this code.

- generate_flowlines.py
  Code used to generate flowlines using the OGGM library. I still don't understand OGGM very well, the code is quite dirty as of right now.

- plot_maker.py
  Code used to generate many many plots for the report and more. The code uses the pyGM library a lot. Produced figures are organised in the imgs folder

- qgis_functions.py
  Precursor to pyGM_funcs.py

- treat_kaskawulsh_files.py
  Code used to manipulate the kaskawulsh data.
 
 - velocity_analysis.py
  THIS CODE IS LOCATED IN THE VELOCITY_ANALYSIS FOLDER. Code used to analyze raster data along glacier flowlines. 

Folders:
- alpha_compare
  Plots produced by compare_alpha.py
  
- imgs
  Many plots produced for the glacier test cases
  
- velocity_analysis
  Plots and data generated from velocity_analysis.py. About surface velocity and dhdt data along flowlines for Little Kluane Glacier.
  
- report
  .tex and .pdf files for the main report. Lots of information about the process in there.
 
- old 
  Code and plots not referenced in the report. Lots of stuff in there were first investigated but deemed not important for the study. See the information (README.txt) in there.
