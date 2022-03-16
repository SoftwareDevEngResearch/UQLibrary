# UQLibrary Course Project README
Thank you for looking at UQLibrary. UQLibrary is a package for sensitivity and identifiability analysis using methods such as parameter subset, active subspace, Morris, and Sobol analyses.

## Set Up and Using *UQLibrary*


## Module Guide
UQLibrary is seperated into three modules; *UQtoolbox*, *lsa*, and *gsa*.

### *UQtoolbox*

The *UQtoolbox* model contains functions *run_uq* for running all package modules and the functions *print_results* and *plot_gsa* for printing, saving, and plotting data results in standardize format. The *UQtoolbox* module defines the three primary classes used to interact with UQLibrary; *model, options,* and *results*. An object of the *model* class holds all information about the system UQLibrary is testing. Fields that are always required are; *evalFcn*, the function mapping from the parameters of interest (POIs) to the quantities of interestes (QOIs), and *basePOIs*, the parameters sensitivity analysis is focused on. If using global methods, *paramDist*, the sampling distributions of each parameter, are required but, if no distributions are provided, *UQLibrary* assumes uniform distributions &pm20;% about the *basePOIs* values. The *options* class holds the subclasses *lsaOptions* and *gsaOptions*, both defined in their respective modules, along with whether to display or plot results and the locations to save results to. The *results* class is the output of the *RunUQ* function and holds the subclasses *lsaResults* and *gsaResults*, both defined in their respective modules.

### *lsa*

The *lsa* module
