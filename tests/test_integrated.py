# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:54:21 2022

@author: USER
"""


import numpy as np
import sys

#Load Load functions to be tested
sys.path.insert(0, '../')
import UQtoolbox_examples
import UQtoolbox as uq

#==============================================================================
#-------------------------------LSA integrated tests---------------------------
#==============================================================================

#----------------------------------Jacobian ----------------------------------
def test_linear_jac_finite():
    (model, options) = UQtoolbox_examples.GetExample("linear")
    #Run only jacobian approximation with finite approx
    options.gsa.run = False
    options.lsa.run_param_subset = False
    options.lsa.x_delta = 1e-3
    options.lsa.method = "finite"
    options.display = False
    
    model.base_poi = np.array([.5, .5])#np.random.uniform(size=2)
    
    results = uq.run_uq(model, options)
    
    true_jac = np.array([1, 2])
    
    assert np.allclose(results.lsa.jac, true_jac)
    
def test_linear_jac_complex():
    (model, options) = UQtoolbox_examples.GetExample("linear")
    #Run only jacobian approximation with finite approx
    options.gsa.run = False
    options.lsa.run_param_subset = False
    options.lsa.x_delta = 1e-16
    options.lsa.method = "complex"
    options.display = False
    
    model.base_poi = np.random.uniform(size=2)
    
    results = uq.run_uq(model, options)
    
    true_jac = np.array([1, 2])
    assert np.allclose(results.lsa.jac, true_jac)
    
#==============================================================================
#------------------------------Identifiability tests---------------------------
#==============================================================================

# Heated Rod
def test_heat_ident_subset_finite():
    (model, options) = UQtoolbox_examples.GetExample("aluminum rod (uniform)")
    #Run only lsa with finite approx
    options.gsa.run = False
    options.lsa.x_delta = 1e-8
    options.lsa.method = "finite"
    options.display = False
    options.lsa.run_param_subset =True
    
    results = uq.run_uq(model, options)
    true_subset = np.array(["h"])
    assert np.all(true_subset==results.lsa.active_set)
    
def test_heat_ident_subset_complex():
    (model, options) = UQtoolbox_examples.GetExample("aluminum rod (uniform)")
    #Run only lsa with complex approx
    options.gsa.run = False
    options.lsa.x_delta = 1e-16
    options.lsa.method = "complex"
    options.display = False
    options.lsa.run_param_subset =True
    
    results = uq.run_uq(model, options)
    true_subset = np.array(["h"])
    assert np.all(true_subset==results.lsa.active_set)
    
# Helmholtz
# def test_helmholtz_ident_subset_complex():
#     (model, options) = UQtoolbox_examples.GetExample("helmholtz")
#     #Run only lsa with finite approx
#     options.gsa.run = False
#     options.lsa.x_delta = 1e-16
#     options.lsa.method = "complex"
#     options.display = False
#     options.lsa.run_param_subset =True
    
#     results = uq.run_uq(model, options)
#     true_subset = np.array(["alpha1", "alpha11", "alpha111"])
#     assert np.all(true_subset==results.lsa.active_set)

#==============================================================================
#-------------------------------GSA integrated tests---------------------------
#==============================================================================