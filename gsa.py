# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:03:35 2022

@author: USER
"""

#3rd party Modules
import numpy as np
#import sys
#import warnings
#import matplotlib.pyplot as plt
#import scipy.integrate as integrate
#from tabulate import tabulate                       #Used for printing tables to terminal
#import sobol                                        #Used for generating sobol sequences
from SALib.sample.sobol_sequence import sample as sobol_sample
import scipy.stats as sct

class GsaOptions:
    def __init__(self, run = True, run_sobol=True, run_morris=True, n_samp_sobol=100000, \
                 n_samp_morris=4, l_morris=3):
        self.run = run
        if self.run == False:
            self.run_sobol = False
            self.run_morris = False
        else:
            self.run_sobol=run_sobol                            #Whether to run Sobol (True or False)
            self.run_morris=run_morris                          #Whether to run Morris (True or False)
        self.n_samp_sobol = n_samp_sobol                      #Number of samples to be generated for GSA
        self.n_samp_morris = n_samp_morris
        self.l_morris=l_morris
        pass

class GsaResults:
    #
    def __init__(self,sobol_base=np.empty, sobol_tot=np.empty, f_a=np.empty, f_b=np.empty, f_d=np.empty, f_ab=np.empty, \
                 samp_d=np.empty,sigma2=np.empty, mu_star=np.empty):
        self.sobol_base=sobol_base
        self.sobol_tot=sobol_tot
        self.f_a=f_a
        self.f_b=f_b
        self.f_d=f_d
        self.f_ab=f_ab
        self.samp_d=samp_d
        self.mu_star=mu_star
        self.sigma2=sigma2
    pass



##--------------------------------------GSA-----------------------------------------------------
def run_gsa(model, gsa_options):
    """Implements global sensitivity analysis using Morris or Sobol analysis.
    
    Parameters
    ----------
    model : Model
        Contaings simulation information.
    gsa_options : GSAOptions
        Contains run settings
        
    Returns
    -------
    GsaResults 
        Holds all run results
    """
    #GSA implements the following local sensitivity analysis methods on "model" object
        # 1) Gets sampling distribution (used only for internal calculations)
        # 2) Calculates Sobol Indices
        # 3) Performs Morris Screenings (not yet implemented)
        # 4) Produces histogram plots for QOI values (not yet implemented)
    # Required Inputs: Object of class "model" and object of class "options"
    # Outputs: Object of class gsa with fisher and sobol elements

    #Get Parameter Distributions
    model=get_samp_dist(model, gsa_options)
    gsa_results = GsaResults()
    #Morris Screening
    if gsa_options.run_morris:
        mu_star, sigma2 = calculate_morris(model, gsa_options)
        gsa_results.mu_star=mu_star
        gsa_results.sigma2=sigma2

    #Sobol Analysis
    if gsa_options.run_sobol:
        #Make Distribution Samples and Calculate model results
        [f_a, f_b, f_ab, f_d, samp_d] = get_samples(model, gsa_options)
        #Calculate Sobol Indices
        [sobol_base, sobol_tot]=calculate_sobol(f_a, f_b, f_ab, f_d)
        gsa_results.f_d=f_d
        gsa_results.f_a=f_a
        gsa_results.f_b=f_b
        gsa_results.f_ab=f_ab
        gsa_results.samp_d=samp_d
        gsa_results.sobol_base=sobol_base
        gsa_results.sobol_tot=sobol_tot
        
    return gsa_results


###----------------------------------------------------------------------------------------------
###-------------------------------------Support Functions----------------------------------------
###----------------------------------------------------------------------------------------------


def get_samples(model,gsa_options):
    """Constructs and evaluates sobol samples using predefined sampling distributions.
        Currently only function for uniform or saltelli normal
    
    Parameters
    ----------
    model : Model
        Contaings simulation information.
    gsa_options : GSAOptions
        Contains run settings
        
    Returns
    -------
    np.ndarray
        n_samp_sobol x n_qoi array of evaluations of Sobol sample part a
    np.ndarray
        n_samp_sobol x n_qoi array of evaluations of Sobol sample part b
    np.ndarray
        n_samp_sobol x n_qoi array x n_poi array of evaluations of mixed Sobol sample ab
    np.ndarray
        2*n_samp_sobol x n_qoi array of concatenated evaluations of part a and b
    np.ndarray
        2*n_samp_sobol x n_poi array of concatenated POI samples of part a and b
    """
    n_samp_sobol = gsa_options.n_samp_sobol
    # Make 2 POI sample matrices with n_samp_sobol samples each
    if model.dist.lower()=='uniform' or model.dist.lower()=='saltellinormal':
        (samp_a, samp_b)=model.samp_dist(n_samp_sobol);                                     #Get both A and B samples so no repeated values
    else:
        #-------Error, need to switch these samplings to the join Saltelli-------------
        samp_a = model.samp_dist(n_samp_sobol)
        samp_b = model.samp_dist(n_samp_sobol)
    # Calculate matrices of QOI values for each POI sample matrix
    f_a = model.eval_fcn(samp_a).reshape([n_samp_sobol, model.n_qoi])  # n_samp_sobol x nQOI out matrix from A
    f_b = model.eval_fcn(samp_b).reshape([n_samp_sobol, model.n_qoi])  # n_samp_sobol x nQOI out matrix from B
    # Stack the output matrices into a single matrix
    f_d = np.concatenate((f_a.copy(), f_b.copy()), axis=0)

    # Initialize combined QOI sample matrices
    if model.n_qoi == 1:
        f_ab = np.empty([n_samp_sobol, model.n_poi])
    else:
        f_ab = np.empty([n_samp_sobol, model.n_poi, model.n_qoi])
    for i_param in range(0, model.n_poi):
        # Define sampC to be A with the ith parameter in B
        samp_ab = samp_a.copy()
        samp_ab[:, i_param] = samp_b[:, i_param].copy()
        if model.n_qoi == 1:
            f_ab[:, i_param] = model.eval_fcn(samp_ab)
        else:
            f_ab[:, i_param, :] = model.eval_fcn(samp_ab)  # n_samp_sobol x nPOI x nQOI tensor
        del samp_ab
    return f_a, f_b, f_ab, f_d, np.concatenate((samp_a.copy(), samp_b.copy()), axis=0)

def calculate_sobol(f_a, f_b, f_ab, f_d):
    """Calculates 1st order and total sobol indices using Saltelli approximation formula.
    
    Parameters
    ----------
    f_a : np.ndarray
        n_samp_sobol x n_qoi array of evaluations of Sobol sample part a
    f_b : np.ndarray
        n_samp_sobol x n_qoi array of evaluations of Sobol sample part b
    f_ab : np.ndarray
        n_samp_sobol x n_qoi array x n_poi array of evaluations of mixed Sobol sample ab
    f_d : np.ndarray
        2*n_samp_sobol x n_qoi array of concatenated evaluations of part a and b
    
        
    Returns
    -------
    np.ndarray
        n_qoi x n_poi array of 1st order Sobol indices
    np.ndarray
        n_qoi x n_poi array of total Sobol indices
    """
    #Calculates calculates sobol indices using satelli approximation method
    #Inputs: model object (with eval_fcn, samp_dist, and nParams)
    #        sobolOptions object
    #Determing number of samples, QOIs, and POIs based on inputs
    if f_ab.ndim==1:
        n_qoi=1
        n_poi=1
    elif f_ab.ndim==2:
        n_qoi=1
        n_poi=f_ab.shape[1]
    elif f_ab.ndim==3:
        n_poi=f_ab.shape[1]
        n_qoi=f_ab.shape[2]
    else:
        raise(Exception('f_ab has greater than 3 dimensions, make sure f_ab is' \
                        'the squeezed form of n_samp_sobol x nPOI x nQOI'))
    #QOI variance
    fDvar=np.var(f_d, axis=0)

    sobol_base=np.empty((n_qoi, n_poi))
    sobol_tot=np.empty((n_qoi, n_poi))
    if n_qoi==1:
        #Calculate 1st order parameter effects
        sobol_base=np.mean(f_b*(f_ab-f_a), axis=0)/(fDvar)

        #Caclulate 2nd order parameter effects
        sobol_tot=np.mean((f_a-f_ab)**2, axis=0)/(2*fDvar)

    else:
        for iQOI in range(0,n_qoi):
            #Calculate 1st order parameter effects
            sobol_base[iQOI,:]=np.mean(f_b[:,[iQOI]]*(f_ab[:,:,iQOI]-f_a[:,[iQOI]]),axis=0)/fDvar[iQOI]
            #Caclulate 2nd order parameter effects
            sobol_tot[iQOI,:]= np.mean((f_a[:,[iQOI]]-f_ab[:,:,iQOI])**2,axis=0)/(2*fDvar[iQOI])


    return sobol_base, sobol_tot

##-------------------------------------GetMorris-------------------------------------------------------
def calculate_morris(model,gsa_options):
    """Calculates morris samples using information from Model and GsaOptions objects.
    
    Parameters
    ----------
    model : Model
        Contaings simulation information.
    gsa_options : GSAOptions
        Contains run settings
        
    Returns
    -------
    np.ndarray
        n_qoi x n_poi array of morris sensitivity mean indices
    np.ndarray
        n_qoi x n_poi array of morris sensitivity variance indices
    """
    #Define delta
    delta=(gsa_options.l_morris+1)/(2*gsa_options.l_morris)
    #Get Parameter Samples- use parameter distribution
    param_samp=model.dist(gsa_options.n_samp_morris)[0]
    #Calulate derivative indices
    d= np.empty((gsa_options.n_samp_morris, model.n_poi, model.n_qoi)) #n_qoi x n_poi x nSamples
    #Define constant sampling matrices
    J=np.ones((model.n_poi+1,model.n_poi))
    B = (np.tril(np.ones(J.shape), -1))
    for i_samp in range(0,gsa_options.n_samp_morris):
        #Define Random Sampling matrices
        D=np.diag(np.random.choice(np.array([1,-1]), size=(model.n_poi,)))
        P=np.identity(model.n_poi)
        #np.random.shuffle(P)
        jTheta=param_samp[i_samp,]*J
        #CalculateMorris Sample matrix
        Bj=np.matmul(jTheta+delta/2*(np.matmul((2*B-J),D)+J),P)
        fBj=model.eval_fcn(Bj)
        for i_poi in np.arange(0,model.n_poi):
            index_nonzero=np.nonzero(Bj[i_poi+1,:]-Bj[i_poi,:])[0][0]
            print(np.nonzero(Bj[i_poi+1,:]-Bj[i_poi,:]))
            if Bj[i_poi+1,index_nonzero]-Bj[i_poi,index_nonzero]>0:
                if model.n_qoi==1:
                    d[i_samp,index_nonzero]=(fBj[i_poi+1]-fBj[i_poi])/delta
                else:
                    d[i_samp,index_nonzero,:]=(fBj[i_poi+1]-fBj[i_poi])/delta
            elif Bj[i_poi+1,index_nonzero]-Bj[i_poi,index_nonzero]<0:
                if model.n_qoi==1:
                    d[i_samp,index_nonzero]=(fBj[i_poi]-fBj[i_poi+1])/delta
                else:
                    d[i_samp,index_nonzero,:]=(fBj[i_poi,:]-fBj[i_poi+1,:])/delta
            else:
                raise(Exception('0 difference identified in Morris'))
    #Compute Indices- all outputs are n_qoi x n_poi
    mu_star=np.mean(np.abs(d),axis=0)
    sigma2=np.var(d, axis=0)

    return mu_star, sigma2


##--------------------------------------GetSampDist----------------------------------------------------
def get_samp_dist(model, gsa_options):
    """Adds sampling function samp_dist to model for drawing of low-discrepency
        from given distribution type.
    
    Parameters
    ----------
    model : Model
        Contaings simulation information.
    gsa_options : GSAOptions
        Contains run settings
        
    Returns
    -------
    Model
        model object with added samp_dist function
    """
    # Determine Sample Function- Currently only 1 distribution type can be defined for all parameters
    if model.dist.lower() == 'normal':  # Normal Distribution
        samp_dist = lambda n_samp_sobol: np.random.randn(n_samp_sobol,model.n_poi)*np.sqrt(model.dist_param[[1], :]) + model.dist_param[[0], :]
    elif model.dist.lower() == 'saltellinormal':
        samp_dist = lambda n_samp_sobol: saltelli_normal(n_samp_sobol, model.dist_param)
    elif model.dist.lower() == 'uniform':  # uniform distribution
        # doubleParms=np.concatenate(model.dist_param, model.dist_param, axis=1)
        samp_dist = lambda n_samp_sobol: saltelli_sample(n_samp_sobol,model.dist_param)
    elif model.dist.lower() == 'exponential': # exponential distribution
        samp_dist = lambda n_samp_sobol: np.random.exponential(model.dist_param,size=(n_samp_sobol,model.n_poi))
    elif model.dist.lower() == 'beta': # beta distribution
        samp_dist = lambda n_samp_sobol:np.random.beta(model.dist_param[[0],:], model.dist_param[[1],:],\
                                               size=(n_samp_sobol,model.n_poi))
    elif model.dist.lower() == 'InverseCDF': #Arbitrary distribution given by inverse cdf
        samp_dist = lambda n_samp_sobol: gsa_options.fcn_inverse_cdf(np.random.rand(n_samp_sobol,model.n_poi))
    else:
        raise Exception("Invalid value for gsa_options.dist. Supported distributions are normal, uniform, exponential, beta, \
        and InverseCDF")  # Raise Exception if invalide distribution is entered
    model.samp_dist=samp_dist
    return model



def saltelli_sample(n_samp_sobol,dist_param):
    """Constructs a uniform low discrepency saltelli sample for use in Sobol
        index approximation
    
    Parameters
    ----------
    n_samp_sobol : int
        Number of samples to take
    dist_param : np.ndarray
        2 x n_poi array of min and max sampling bounds for each parameter
        
    Returns
    -------
    np.ndarray
        POI sample part a
    np.ndarray
        POI sample part b
    """
    n_poi=dist_param.shape[1]
    #base_sample=sobol.sample(dimension=n_poi*2, n_points=n_samp_sobol, skip=1099)
    base_sample=sobol_sample(n_samp_sobol,n_poi*2)
    base_a=base_sample[:,:n_poi]
    base_b=base_sample[:,n_poi:2*n_poi]
    samp_a=dist_param[[0],:]+(dist_param[[1],:]-dist_param[[0],:])*base_a
    samp_b=dist_param[[0],:]+(dist_param[[1],:]-dist_param[[0],:])*base_b
    return (samp_a, samp_b)

def saltelli_normal(n_samp_sobol, dist_param):
    """Constructs a normal low discrepency saltelli sample for use in Sobol
        index approximation
    
    Parameters
    ----------
    n_samp_sobol : int
        Number of samples to take
    dist_param : np.ndarray
        2 x n_poi array of mean and variance for each parameter
        
    Returns
    -------
    np.ndarray
        POI sample part a
    np.ndarray
        POI sample part b
    """
    n_poi=dist_param.shape[1]
    #base_sample=sobol.sample(dimension=n_poi*2, n_points=n_samp_sobol, skip=1099)
    base_sample=sobol_sample(n_samp_sobol,n_poi*2)
    base_a=base_sample[:,:n_poi]
    base_b=base_sample[:,n_poi:2*n_poi]
    transformA=sct.norm.ppf(base_a)
    transformB=sct.norm.ppf(base_b)
    samp_a=transformA*np.sqrt(dist_param[[1], :]) + dist_param[[0], :]
    samp_b=transformB*np.sqrt(dist_param[[1], :]) + dist_param[[0], :]
    return (samp_a, samp_b)
