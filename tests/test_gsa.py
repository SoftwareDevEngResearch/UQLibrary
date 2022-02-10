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
from gsa import saltelli_sample
from gsa import get_samp_dist

#==============================================================================
#------------------------------Morris Sampling---------------------------------
#==============================================================================

#Check to confirm only a single value is perturbed for each 
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
    sample_dif_sum = np.sum(sample_dif, axis = 1)
    assert np.allclose(sample_dif_sum, (1/3*np.ones(sample_dif_sum.shape)))

#Check to confirm the morris sample has no repeated values
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
    #Make all combinations False to ensure every value is checked
    sample_diff = np.zeros((n_poi+1, n_poi+1), dtype=bool)
    for i_samp1 in range(n_poi+1):
        for i_samp2 in range(n_poi+1):
            #Fix same samples to be true since they should be equal
            if i_samp1==i_samp2:
                sample_diff[i_samp1, i_samp2] = True
            #Check every different sample is actually different
            elif np.any(sample[i_samp1,:]!=sample[i_samp2,:]):
                sample_diff[i_samp1,i_samp2] = True
            
            
    assert np.all(sample_diff)

#==============================================================================
#-------------------------Distribution Sampling--------------------------------
#==============================================================================

# 1)======================satelli_sample test==================================
# 1a) Check low discrepancy sampling has no repeated values
def test_satelli_repeats():
    n_samp = 6
    n_poi = 2
    sample = saltelli_sample(n_samp, n_poi)
    sample_diff = np.zeros((n_samp, n_samp), dtype=bool)
    for i_samp1 in range(n_samp):
        for i_samp2 in range(n_samp):
            #Fix same samples to be true since they should be equal
            if i_samp1==i_samp2:
                sample_diff[i_samp1, i_samp2] = True
            #Check every different sample is actually different
            elif np.any(sample[i_samp1,:]!=sample[i_samp2,:]):
                sample_diff[i_samp1,i_samp2] = True
    print(sample)
    assert np.all(sample_diff)
    
# 1b) Check sample is entirely between [0,1]
def test_satelli_ranges():
    n_samp = 1000
    n_poi = 20
    sample = saltelli_sample(n_samp, n_poi)
    
    assert np.min(sample) >= 0 and np.max(sample) <= 1
    
# 1c) Check mean is .5 for each poi
def test_satelli_mean():
    n_samp = 1000000
    n_poi = 7
    sample = saltelli_sample(n_samp, n_poi)
    
    assert np.allclose(np.mean(sample, axis = 0), .5, rtol = 1e-3, atol = 1e-3)
    
# 1d) Check variance is 1/12 for each poi
def test_satelli_var():
    n_samp = 1000000
    n_poi = 3
    sample = saltelli_sample(n_samp, n_poi)
    
    assert np.allclose(np.var(sample, axis = 0), 1/12, rtol = 1e-3, atol = 1e-3)
# 2)=========================saltelli_uniform test=============================
    
# 2a) Check mean is (b-a)/2 for each poi
def test_satelli_uniform_mean():
    n_samp = 1000000
    n_poi = 4
    dist_param = np.random.rand(2,n_poi)
    #Sort so that a<=b
    dist_param = np.sort(dist_param, axis = 1)
    sample_fcn = get_samp_dist("saltelli uniform", dist_param, n_poi)
    sample = sample_fcn(n_samp)
    
    expected_mean = (dist_param[1,:]+dist_param[0,:])/2
    
    assert np.allclose(np.mean(sample, axis = 0), expected_mean , rtol = 1e-3, atol = 1e-3)
    
# 2b) Check variance is (b-a)**2/12 for each poi
def test_satelli_uniform_var():
    n_samp = 1000000
    n_poi = 5
    dist_param = np.random.rand(2,n_poi)
    #Sort so that a<=b
    dist_param = np.sort(dist_param, axis = 0)
    sample_fcn = get_samp_dist("saltelli uniform", dist_param, n_poi)
    sample = sample_fcn(n_samp)
    
    expected_var = ((dist_param[1,:]-dist_param[0,:])**2)/12
    true_var = np.var(sample, axis = 0)
    print("Distribution parameters: " + str(dist_param))
    print(dist_param[1,:])
    print(dist_param[0,:])
    print("Expected variances: " + str(expected_var))
    print("True variances: " + str(true_var))
    
    assert np.allclose(true_var, expected_var , rtol = 1e-3, atol = 1e-3)
    
# 2c) Check all samples in [a,b]
def test_saltelli_uniform_ranges():
    n_samp = 1000000
    n_poi = 5
    dist_param = np.random.rand(2,n_poi)
    #Sort so that a<=b
    dist_param = np.sort(dist_param, axis = 0)
    sample_fcn = get_samp_dist("saltelli uniform", dist_param, n_poi)
    sample = sample_fcn(n_samp)
    
    expected_min = dist_param[0,:]
    expected_max = dist_param[1,:]
    true_min = np.min(sample, axis =0)
    true_max = np.max(sample, axis =0)
    
    assert np.all(expected_min<=true_min) and np.all(expected_max>=true_max)
    
# 3)=========================saltelli_normal test=============================
    
# 3a) Check mean is mu for each poi
def test_satelli_normal_mean():
    n_samp = 1000000
    n_poi = 3
    dist_param = np.random.rand(2,n_poi)
    #Sort so that a<=b
    dist_param = np.sort(dist_param, axis = 1)
    sample_fcn = get_samp_dist("saltelli normal", dist_param, n_poi)
    sample = sample_fcn(n_samp)
    
    expected_mean = dist_param[0,:]
    true_mean = np.mean(sample, axis = 0)
    
    assert np.allclose(true_mean, expected_mean , rtol = 1e-3, atol = 1e-3)
    
# 3b) Check variance is sigma**2 for each poi
def test_satelli_uniform_var():
    n_samp = 1000000
    n_poi = 6
    dist_param = np.random.rand(2,n_poi)
    #Sort so that a<=b
    dist_param = np.sort(dist_param, axis = 0)
    sample_fcn = get_samp_dist("saltelli normal", dist_param, n_poi)
    sample = sample_fcn(n_samp)
    
    expected_var = dist_param[1,:]
    true_var = np.var(sample, axis = 0)
    
    print("Distribution parameters: " + str(dist_param))
    print("Expected variances: " + str(expected_var))
    print("True variances: " + str(true_var))
    
    assert np.allclose(true_var, expected_var , rtol = 1e-3, atol = 1e-3)
    
# 4)=============================uniform test==================================
    
# 4a) Check mean is (b-a)/2 for each poi
def test_uniform_mean():
    n_samp = 1000000
    n_poi = 4
    dist_param = np.random.rand(2,n_poi)
    #Sort so that a<=b
    dist_param = np.sort(dist_param, axis = 1)
    sample_fcn = get_samp_dist("uniform", dist_param, n_poi)
    sample = sample_fcn(n_samp)
    
    expected_mean = (dist_param[1,:]+dist_param[0,:])/2
    
    assert np.allclose(np.mean(sample, axis = 0), expected_mean , rtol = 1e-3, atol = 1e-3)
    
# 4b) Check variance is (b-a)**2/12 for each poi
def test_uniform_var():
    n_samp = 1000000
    n_poi = 5
    dist_param = np.random.rand(2,n_poi)
    #Sort so that a<=b
    dist_param = np.sort(dist_param, axis = 0)
    sample_fcn = get_samp_dist("uniform", dist_param, n_poi)
    sample = sample_fcn(n_samp)
    
    expected_var = ((dist_param[1,:]-dist_param[0,:])**2)/12
    true_var = np.var(sample, axis = 0)
    print("Distribution parameters: " + str(dist_param))
    print(dist_param[1,:])
    print(dist_param[0,:])
    print("Expected variances: " + str(expected_var))
    print("True variances: " + str(true_var))
    
    assert np.allclose(true_var, expected_var , rtol = 1e-3, atol = 1e-3)
    
# 4c) Check all samples in [a,b]
def test_uniform_ranges():
    n_samp = 1000000
    n_poi = 5
    dist_param = np.random.rand(2,n_poi)
    #Sort so that a<=b
    dist_param = np.sort(dist_param, axis = 0)
    sample_fcn = get_samp_dist("uniform", dist_param, n_poi)
    sample = sample_fcn(n_samp)
    
    expected_min = dist_param[0,:]
    expected_max = dist_param[1,:]
    true_min = np.min(sample, axis =0)
    true_max = np.max(sample, axis =0)
    
    assert np.all(expected_min<=true_min) and np.all(expected_max>=true_max)
    
# 5)=============================normal test===================================
    
# 5a) Check mean is mu for each poi
def test_normal_mean():
    n_samp = 1000000
    n_poi = 3
    dist_param = np.random.rand(2,n_poi)
    #Sort so that a<=b
    dist_param = np.sort(dist_param, axis = 1)
    sample_fcn = get_samp_dist("normal", dist_param, n_poi)
    sample = sample_fcn(n_samp)
    
    expected_mean = dist_param[0,:]
    true_mean = np.mean(sample, axis = 0)
    
    assert np.allclose(true_mean, expected_mean , rtol = 1e-3, atol = 1e-3)
    
# 5b) Check variance is sigma**2 for each poi
def test_satelli_uniform_var():
    n_samp = 1000000
    n_poi = 6
    dist_param = np.random.rand(2,n_poi)
    #Sort so that a<=b
    dist_param = np.sort(dist_param, axis = 0)
    sample_fcn = get_samp_dist("normal", dist_param, n_poi)
    sample = sample_fcn(n_samp)
    
    expected_var = dist_param[1,:]
    true_var = np.var(sample, axis = 0)
    
    print("Distribution parameters: " + str(dist_param))
    print("Expected variances: " + str(expected_var))
    print("True variances: " + str(true_var))
    
    assert np.allclose(true_var, expected_var , rtol = 1e-3, atol = 1e-3)
    
    