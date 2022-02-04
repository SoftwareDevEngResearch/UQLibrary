#UQtoolbox
#Module for calculating local sensitivity indices, parameter correlation, and Sobol indices for an arbitrary model
#Authors: Harley Hanes, NCSU, hhanes@ncsu.edu
#Required Modules: numpy, seaborne
#Functions: LSA->GetJacobian
#           GSA->ParamSample, PlotGSA, GetSobol

#3rd party Modules
import numpy as np
import sys
import warnings
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from tabulate import tabulate                       #Used for printing tables to terminal
#import sobol                                        #Used for generating sobol sequences
import SALib.sample.saltelli as sobol
import scipy.stats as sct

#Package Modules
import lsa
import gsa
#import seaborne as seaborne
###----------------------------------------------------------------------------------------------
###-------------------------------------Class Definitions----------------------------------------
###----------------------------------------------------------------------------------------------

##--------------------------------------uqOptions--------------------------------------------------
#Define class "uqOptions", this will be the class used to collect algorithm options for functions
#   -Subclasses: lsaOptions, plotOptions, gsaOptions
#--------------------------------------lsaOptions------------------------------------------------

#--------------------------------------gsaOptions------------------------------------------------

#--------------------------------------plotOptions------------------------------------------------
class PlotOptions:
    def __init__(self,run=True,n_points=400,path=False):
        self.run=run
        self.n_points=n_points
        self.path=path
        pass
#--------------------------------------uqOptions------------------------------------------------
#   Class holding the above options subclasses
class Options:
    def __init__(self,lsa=lsa.LsaOptions(),plot=PlotOptions(),gsa=gsa.GsaOptions(), \
                 display=True, save=False, path='..'):
        self.lsa=lsa
        self.plot=plot
        self.gsa=gsa
        self.display=display                       #Whether to print results to terminal
        self.save=save                             #Whether to save results to files
        self.path=path                             #Where to save files
        if self.save and not self.path:
            warnings.warn("Save marked as true but no path given, saving files to current folder.")
            path=''
    pass

##-------------------------------------model------------------------------------------------------
#Define class "model", this will be the class used to collect input information for all functions
class Model:
    #Model sets should be initialized with base parameter settings, covariance Matrix, and eval function that
    #   takes in a vector of POIs and outputs a vector of QOIs
    def __init__(self,base_poi=np.empty(0), name_poi = np.empty(0), \
                 name_qoi= np. empty(0), cov=np.empty(0), \
                 eval_fcn=np.empty(0), dist='unif', dist_param="auto"):
        self.base_poi=base_poi
        if not isinstance(self.base_poi,np.ndarray):                    #Confirm that base_poi is a numpy array
            warnings.warn("model.base_poi is not a numpy array")
        if np.ndim(self.base_poi)>1:                                    #Check to see if base_poi is a vector
            self.base_poi=np.squeeze(self.base_poi)                     #Make a vector if an array with 1 dim greater than 1
            if np.ndim(self.base_poi)!=1:                               #Issue an error if base_poi is a matrix or tensor
                raise Exception("Error! More than one dimension of size 1 detected for model.base_poi, model.base_poi must be dimension 1")
            else:                                                       #Issue a warning if dimensions were squeezed out of base POIs
                warnings.warn("model.base_poi was reduced a dimension 1 array. No entries were deleted.")
        self.n_poi=self.base_poi.size
        #Assign name_poi
        self.name_poi = name_poi                                            #Assign name_poi called
        if (self.name_poi.size != self.n_poi) & (self.name_poi.size !=0):   #Check that correct size if given
            warnings.warn("name_poi entered but the number of names does not match the number of POIs. Ignoring names.")
            self.name_poi=np.empty(0)
        if self.name_poi.size==0:                                           #If not given or incorrect size, number POIs
            POInumbers=np.arange(0,self.n_poi)
            self.name_poi=np.char.add('POI',POInumbers.astype('U'))
        #Assign evaluation function and compute base_qoi
        self.eval_fcn=eval_fcn
        self.base_qoi=eval_fcn(base_poi)
        if not isinstance(self.base_qoi,np.ndarray):                    #Confirm that base_qoi is a numpy array
            warnings.warn("model.base_qoi is not a numpy array")
        print(self.base_qoi)
        self.n_qoi=len(self.base_qoi)
        #Assign QOI names
        self.name_qoi = name_qoi
        if (self.name_qoi.size !=self.n_qoi) & (self.name_qoi.size !=0):    #Check names if given match number of QOIs
            warnings.warn("name_qoi entered but the number of names does not match the number of QOIs. Ignoring names.")
            self.name_qoi = np.empty(0)
        if self.name_qoi.size==0:                                 #If not given or incorrect size, number QOIs
            QOInumbers = np.arange(0, self.n_qoi)
            self.name_qoi = np.char.add('QOI', QOInumbers.astype('U'))
        #Assign covariance matrix
        self.cov=cov
        if self.cov.size!=0 and np.shape(self.cov)!=(self.n_poi,self.n_poi): #Check correct sizing
            raise Exception("Error! model.cov is not an nPOI x nPOI array")
        
        #Assign distributions
        self.dist = dist   
        
        #Parse through possible options for dist_param: 
        # 1) False (use base settings)
        # 2) numpy array (check dimensions align with n_pois and dist requirements)
        # Otherwise raise exception
        if str(dist_param).lower() == 'auto':
            if self.dist.lower()=='uniform':
                self.dist_param=[[.8],[1.2]]*np.ones((2,self.n_poi))*self.base_poi
            elif self.dist.lower()=='normal':
                if cov.size()==0:
                    self.dist_param=[[1],[.2]]*np.ones((2,self.n_poi))*self.base_poi
                else:
                    self.dist_param=[self.base_poi, np.diag(self.cov,k=0)]
            elif dist_param.lower() == 'null':
                self.dist_param = dist_param
            else:
                raise Exception("Unrecognized entry for dist_param: " + str(dist_param))
        elif type(dist_param) == np.ndarray:
            #Check dimensions of numpy array are correct
            if dist_param.shape[1] == self.n_poi:
                # Correct number of parameters for each distribution
                if dist == "unif" and dist_param.shape[0]!=2:
                    raise Exception("2 parameters per POI required for uniform")
                elif dist == "normal" and dist_param.shape[0]!=2:
                    raise Exception("2 parameters per POI required for normal")
                else :
                    self.dist_param=dist_param
            else:
                raise Exception("Incorrect shape of dist_param. Given shape: "\
                                + dist_param.shape + ", desired shape: ... x n_poi")     
        else:
            raise Exception("Incorrect data-type for dist_param, use ndarray or 'auto'")
    # Hidden parameters - parameters that are not set in calling and should not be adjusted
    pass
    def copy(self):
        return Model(base_poi=self.base_poi, name_poi = self.name_poi, name_qoi= self.name_qoi, cov=self.cov, \
                 eval_fcn=self.eval_fcn, dist=self.dist,dist_param=self.dist_param)

##------------------------------------results-----------------------------------------------------
# Define class "results" which holds a gsaResults object and lsaResults object

class Results:
    def __init__(self,lsa=lsa.LsaResults(), gsa=gsa.GsaResults()):
        self.lsa=lsa
        self.gsa=gsa
    pass


###----------------------------------------------------------------------------------------------
###-------------------------------------Main Functions----------------------------------------
###----------------------------------------------------------------------------------------------
#   The following functions are the primary functions for running the package. RunUQ runs both local sensitivity
#   analysis and global sensitivity analysis while printing to command window summary statistics. However, local
#   sensitivity analysis and global sensitivity analysis can be run independently with LSA and GSA respectively

##--------------------------------------RunUQ-----------------------------------------------------
def run_uq(model, options):
    """Runs both the local and global sensitivity, and result printing, saving, and plotting.
    
    Parameters
    ----------
    model : Model
        Object of class Model holding run information.
    options : Options
        Object of class Options holding run settings.
        
    Returns
    -------
    Results 
        Object of class Results holding all run results.
    """
    results = Results()
    #Run Local Sensitivity Analysis
    if options.lsa.run:
        results.lsa = lsa.run_lsa(model, options.lsa)

    #Run Global Sensitivity Analysis
    # if options.gsa.run:
        # if options.lsa.run:
        #     #Use a reduced model if it was caluclated
        #     results.gsa=GSA(results.lsa.reducedModel, options)
        # else:
    if options.gsa.run:
        results.gsa = gsa.run_gsa(model, options.gsa)

    #Print Results
    if options.display:
        print_results(results,model,options)                     #Print results to standard output path

    if options.save:
        original_stdout = sys.stdout                            #Save normal output path
        sys.stdout=open(options.path + 'Results.txt', 'a+')            #Change output path to results file
        print_results(results,model,options)                     #Print results to file
        sys.stdout=original_stdout                              #Revert normal output path

    #Plot Samples
    if options.gsa.run_sobol & options.gsa.run:
        plot_gsa(model, results.gsa.samp_d, results.gsa.f_d, options)

    return results



def print_results(results,model,options):
    """Prints Results object to console or document.
    
    Parameters
    ----------
    results : Results
        Object of class Model holding run information.
    model : Model
        Object of class Model holding run information.
    options : Options
        Object of class Options holding run settings.
    """
    # Print Results
    #Results Header
    print('Sensitivity results for nSampSobol=' + str(options.gsa.n_samp_sobol))
    #Local Sensitivity Analysis
    if options.lsa.run:
        print('\n Base POI Values')
        print(tabulate([model.base_poi], headers=model.name_poi))
        print('\n Base QOI Values')
        print(tabulate([model.base_qoi], headers=model.name_qoi))
        print('\n Sensitivity Indices')
        print(tabulate(np.concatenate((model.name_poi.reshape(model.n_poi,1),np.transpose(results.lsa.jac)),1),
              headers= np.append("",model.name_qoi)))
        print('\n Relative Sensitivity Indices')
        print(tabulate(np.concatenate((model.name_poi.reshape(model.n_poi,1),np.transpose(results.lsa.rsi)),1),
              headers= np.append("",model.name_qoi)))
        #print("Fisher Matrix: " + str(results.lsa.fisher))
        #Active Subsapce Analysis
        print('\n Active Supspace')
        print(results.lsa.active_set)
        print('\n Inactive Supspace')
        print(results.lsa.inactive_set)
    if options.gsa.run: 
        if options.gsa.run_sobol:
            if model.n_qoi==1:
                print('\n Sobol Indices for ' + model.name_qoi[0])
                print(tabulate(np.concatenate((model.name_poi.reshape(model.n_poi,1), results.gsa.sobol_base.reshape(model.n_poi,1), \
                                               results.gsa.sobol_tot.reshape(model.n_poi,1)), 1),
                               headers=["", "1st Order", "Total Sensitivity"]))
            else:
                for iQOI in range(0,model.n_qoi):
                    print('\n Sobol Indices for '+ model.name_qoi[iQOI])
                    print(tabulate(np.concatenate((model.name_poi.reshape(model.n_poi,1),results.gsa.sobol_base[[iQOI],:].reshape(model.n_poi,1), \
                        results.gsa.sobol_tot[[iQOI],:].reshape(model.n_poi,1)),1), headers = ["", "1st Order", "Total Sensitivity"]))
    
        if options.gsa.run_morris:
            if model.n_qoi==1:
                print('\n Morris Screening Results for' + model.name_qoi[0])
                print(tabulate(np.concatenate((model.name_poi.reshape(model.n_poi, 1), results.gsa.mu_star.reshape(model.n_poi, 1), \
                                               results.gsa.sigma2.reshape(model.n_poi, 1)), 1),
                    headers=["", "mu_star", "sigma2"]))
            else:
                print('\n Morris Screening Results for' + model.name_qoi[iQOI])
                print(tabulate(np.concatenate(
                    (model.name_poi.reshape(model.n_poi, 1), results.gsa.mu_star[[iQOI], :].reshape(model.n_poi, 1), \
                     results.gsa.sigma2[[iQOI], :].reshape(model.n_poi, 1)), 1),
                    headers=["", "mu_star", "sigma2"]))

###----------------------------------------------------------------------------------------------
###-------------------------------------Support Functions----------------------------------------
###----------------------------------------------------------------------------------------------


##--------------------------------------GetSobol------------------------------------------------------
# GSA Component Functions


#
#
def plot_gsa(model, sample_mat, eval_mat, options):
    """Plots Sobol Sampling results from gsa module.
    
    Parameters
    ----------
    model : Model
        Object of class Model holding run information.
    sample_mat: np.ndarray
        n_samp x n_poi array holding each parameter sample
    eval_mat : np.ndarray
        n_samp x n_qoi array holding each function evaluation
    options : Options
        Object of class Options holding run settings.
    """
    #Reduce Sample number
    #plotPoints=range(0,int(sample_mat.shape[0]), int(sample_mat.shape[0]/plotOptions.n_points))
    #Make the number of sample points to survey
    plotPoints=np.linspace(start=0, stop=sample_mat.shape[0]-1, num=options.plot.n_points, dtype=int)
    #Plot POI-POI correlation and distributions
    figure, axes=plt.subplots(nrows=model.n_poi, ncols= model.n_poi, squeeze=False)
    for iPOI in range(0,model.n_poi):
        for jPOI in range(0,iPOI+1):
            if iPOI==jPOI:
                n, bins, patches = axes[iPOI, jPOI].hist(sample_mat[:,iPOI], bins=41)
            else:
                axes[iPOI, jPOI].plot(sample_mat[plotPoints,iPOI], sample_mat[plotPoints,jPOI],'b*')
            if jPOI==0:
                axes[iPOI,jPOI].set_ylabel(model.name_poi[iPOI])
            if iPOI==model.n_poi-1:
                axes[iPOI,jPOI].set_xlabel(model.name_poi[jPOI])
            if model.n_poi==1:
                axes[iPOI,jPOI].set_ylabel('Instances')
    figure.tight_layout()
    if options.path:
        plt.savefig(options.path+"POIcorrelation.png")

    #Plot QOI-QOI correlationa and distributions
    figure, axes=plt.subplots(nrows=model.n_qoi, ncols= model.n_qoi, squeeze=False)
    for iQOI in range(0,model.n_qoi):
        for jQOI in range(0,iQOI+1):
            if iQOI==jQOI:
                axes[iQOI, jQOI].hist([eval_mat[:,iQOI]], bins=41)
            else:
                axes[iQOI, jQOI].plot(eval_mat[plotPoints,iQOI], eval_mat[plotPoints,jQOI],'b*')
            if jQOI==0:
                axes[iQOI,jQOI].set_ylabel(model.name_qoi[iQOI])
            if iQOI==model.n_qoi-1:
                axes[iQOI,jQOI].set_xlabel(model.name_qoi[jQOI])
            if model.n_qoi==1:
                axes[iQOI,jQOI].set_ylabel('Instances')
    figure.tight_layout()
    if options.path:
        plt.savefig(options.path+"QOIcorrelation.png")

    #Plot POI-QOI correlation
    figure, axes=plt.subplots(nrows=model.n_qoi, ncols= model.n_poi, squeeze=False)
    for iQOI in range(0,model.n_qoi):
        for jPOI in range(0, model.n_poi):
            axes[iQOI, jPOI].plot(sample_mat[plotPoints,jPOI], eval_mat[plotPoints,iQOI],'b*')
            if jPOI==0:
                axes[iQOI,jPOI].set_ylabel(model.name_qoi[iQOI])
            if iQOI==model.n_qoi-1:
                axes[iQOI,jPOI].set_xlabel(model.name_poi[jPOI])
    if options.path:
        plt.savefig(options.path+"POI_QOIcorrelation.png")
    #Display all figures
    if options.display:
        plt.show()

