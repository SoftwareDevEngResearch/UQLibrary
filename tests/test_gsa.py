# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:45:11 2022

@author: USER
"""



import numpy as np
import sys

#Load Load functions to be tested
sys.path.insert(0, '../')
from gsa import get_morris_poi_sample

def test_morris_sampling_delta():
    #Only let the base sample be 0 in 3D
    n_poi = 3
    param_dist = lambda n_samp : np.zeros((n_samp,n_poi))
    #Use two samples to check repeatability
    n_samp = 1
    #perturbation distance is flexible
    delta = 1/3
    
    sample = get_morris_poi_sample(param_dist, n_samp, n_poi, delta, \
                                  random = True)
    
    #Check sample difference
    sample_dif = sample[1:,:]-sample[0:-1,:]
    sample_dif_sum_abs = np.sum(np.abs(sample_dif), axis = 1)
    assert np.allclose(sample_dif_sum_abs, (1/3*np.ones(sample_dif_sum_abs.shape)))

def test_morris_sampling_repeats():
    #Only let the base sample be 0 in 3D
    n_poi = 3
    param_dist = lambda n_samp : np.zeros((n_samp,n_poi))
    #Use two samples to check repeatability
    n_samp = 1
    #perturbation distance is flexible
    delta = 1/3
    
    sample = get_morris_poi_sample(param_dist, n_samp, n_poi, delta, \
                                  random = True)
    
        
    print(sample)
    sample_diff = np.ones((n_poi+1, n_poi+1), dtype=bool)
    for i_samp1 in range(n_poi+1):
        for i_samp2 in range(n_poi+1):
            if i_samp1==i_samp2:
                sample_diff[i_samp1, i_samp2] = True
            elif np.any(sample[i_samp1,:]!=sample[i_samp2,:]):
                sample_diff[i_samp1,i_samp2] = True
            else :
                sample_diff[i_samp1,i_samp2] = False
            
            
            
    assert np.all(sample_diff)