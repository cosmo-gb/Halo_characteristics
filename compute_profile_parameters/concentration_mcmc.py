#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September/October 2023

@author: guillaume
"""

"""
This script is a draft for fitting halo profile and compute concentration with mcmc method
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import emcee

from semi_analytical_halos.generate_smooth_halo import Smooth_halo


if __name__ == '__main__':
    SMH = Smooth_halo()
    N_part = 10000
    N_bin = 20
    my_halo = SMH.smooth_halo_creation(N_part=N_part, N_bin=N_bin) #kind_profile, b_ax=0.5, c_ax=0.5)
    data = my_halo["data"]
    r_data = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
    print(r_data)
    out = SMH.profile_log_r_bin_hist(r_data, N_bin=N_bin)
    for k in out.keys():
        print(k)

    def model(conc, mass, x_data,):
        print(conc)
        y_th = SMH.profile_NFW(x_data, conc, mass)
        return y_th
    

    """
    def log_likelihood(param, x_data, y_data,):
        print(param)
        conc, sigma = param
        var_2 = 2*sigma**2
        mass = len(x_data)
        residual = np.sum((model(param, mass, x_data) - y_data)**2)/var_2
        out = -np.log(residual) - 0.5 * mass * np.log(var_2*np.pi)
        return out
    """
    
    def log_likelihood(theta, x_data, y_data, yerr):
        conc, log_f = theta
        mass = len(x_data)
        model = SMH.profile_NFW(x_data, theta, mass)
        sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((y_data - model) ** 2 / sigma2 + np.log(sigma2))

    
    def log_prior(param,):
        conc = param[0]
        if conc > 0.01 and conc < 10000:
            return 0.0
        else :     
            return -np.inf

    def log_probability(param, x_data, y_data, y_err,):
        print(param)
        lp = log_prior(param)
        if not np.isfinite(lp):
            return -np.inf
        else :
            return lp + log_likelihood(param, x_data, y_data, y_err) 

    def main(p0, nwalkers, niter, ndim, log_probability, x_data, y_data, y_err,):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x_data, y_data, y_err))

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 100)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter)

        return sampler, pos, prob, state
    
    initial = np.array([3, 1])
    ndim = len(initial)
    nwalkers = 100
    p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]
    niter = 1000
    y_err = out["rho"] / np.sqrt(out["N_part_in_shell"])
    x_err = out["size_shell"]
    sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, log_probability, out["radius"], out["rho"], y_err)

    def do_plot(sampler):
        samples = sampler.get_chain(flat=True)
        print(samples.shape)
        plt.hist(samples[:, 0], 100, color="k", histtype="step")
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$p(\theta_1)$")
        plt.gca().set_yticks([])
        print(np.mean(samples, axis=0), np.median(samples, axis=0), np.std(samples, axis=0))
        plt.show()
    
    do_plot(sampler)


