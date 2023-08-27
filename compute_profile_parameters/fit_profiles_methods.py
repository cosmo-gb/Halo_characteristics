#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:04:45 2022

@author: guillaume
"""



import numpy as np
import scipy.integrate as integrate
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt





def profile_log_r_bin_hist(radius,r_min=0.01,r_max=1,N_bin=30,factor_random_mass=1):
    # It computes the density profile with an equal logarithmic shell size
    # It imposes shells of a given size, computed with r_min r_max and N_bin
    # Then it computes the number of particles in each shell
    # the density is simply N_part/V_shell
    # radius is the particles radii, not need to be sorted
    # radius, r_min and r_max should be in the same length unit
    # N_bin is the number of shells used for the density profile
    # rho_loc should be sorted as radius and in rho_crit unit (optional, to compute the scatter of the profile)
    # factor_random: take into account the mass differnce between a particle mass in the halo 
    # and a particle in the simulation DEUS, it can be =1 if I am using data from DEUS
    # r_bin_log: contains N_bin elements, radius_mean of each shell
    # rho_bin_log: contains N_bin elements, density profile in the shells
    # N_part_in_shell: contains N_bin elements, number of particles in each shell
    if r_min <= 0 :
        print('r_min should be strictly positive')
    r_log_bin = np.logspace(np.log10(r_min),np.log10(r_max),N_bin+1) 
    N_part_in_shell, r_shell = np.histogram(radius, bins=r_log_bin) 
    N_part_not_res = len(np.where(radius < r_min)[0])
    dn, dr = np.zeros((N_bin+1),dtype=int), np.zeros((N_bin+1))
    dn[0] = N_part_not_res
    dn[1:] = N_part_in_shell
    dr[0] = r_log_bin[0]
    dr[1:] = r_log_bin[1:] - r_log_bin[:-1]
    r_log_bin = (r_log_bin[1:] + r_log_bin[:-1])/2
    Volume_shell = (4*np.pi/3)*(r_shell[1:]**3 - r_shell[:-1]**3)
    rho_log_bin = N_part_in_shell*factor_random_mass/Volume_shell
    return(r_log_bin,rho_log_bin,dn,dr) 

def fit_with_abg_profile(radius,res=0.01,N_bin=30):
    # This function has been tested on semi-analytical (Monte-Carlo like) halos
    # with different sphericities, concentrations, slopes and total particle numbers.
    # The typical error is of ~ 30 %, 10 %, 3% for 10**3, 10**4 and 10**5 particles respectively.
    # Typically, the error is slightly less important for the concentration than for the slopes
    r_log_bin, rho_log_bin, N_part_in_shell = profile_log_r_bin_hist(radius,r_min=res,N_bin=N_bin)
    N_vir_sph = len(radius)
    def profile_abc(r_data,concentration,beta,gamma):
        alpha = 1 # I set alpha = 1
        # profile abc: rho(r) = rho_s * ( x**(-gamma) ) * (1 + x**alpha)**((gamma-beta)/alpha), x=r/r_s
        # r_data should be in Mpc/h
        # rho_s should be in rho_crit
        # concentration, alpha, beta, gamma are dimensionless
        mass = N_vir_sph # Number of particles in the halo
        Rvir = 1 # should be in the same dimension than the data, so I should let Rvir=1 and put the data (r_data) in Rvir unit
        r_minus_2 = Rvir/concentration
        # in the range: 0 <= gamma < 2 < beta, the following equation is correct
        r_s = r_minus_2 * ((beta - 2)/(2 - gamma))**(1/alpha) # the scale radius in function of r_minus_2 in a alpha, beta, gamma profile
        prof_abc_x_2 = lambda x: (x**(2-gamma)) * (1 + x**alpha)**((gamma - beta)/alpha)
        my_int = integrate.quad(lambda x: prof_abc_x_2(x),0,Rvir/r_s)
        rho_s = (mass/(4*np.pi*(r_s**3)*my_int[0]))
        x_data = r_data/r_s # dimensionless
        rho = rho_s * ( x_data**(-gamma) ) * (1 + x_data**alpha)**((gamma-beta)/alpha) # in rho_crit
        return(np.log10(rho))
    ####################################################################################
    # No bounds => Levenberg-Marquardt Algorithm: 
    # Implementation and Theory,” Numerical Analysis, ed. G. A. Watson, Lecture Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
    #popt, pcov = curve_fit(profile_abc,r_log_bin,np.log10(rho_log_bin))
    #perr = np.sqrt(np.diag(pcov))
    #print('No bounds: c,beta,gamma =',popt)
    ####################################################################################
    # with bounds => Trust Region reflective
    # M. A. Branch, T. F. Coleman, and Y. Li, “A Subspace, Interior, and Conjugate Gradient Method for Large-Scale Bound-Constrained Minimization Problems,” SIAM Journal on Scientific Computing, Vol. 21, Number 1, pp 1-23, 1999.
    my_bound = ([1,2.1,0.],[100,5,1.9]) # the bounds for the concentration, beta and gamma
    popt, pcov = curve_fit(profile_abc,r_log_bin,np.log10(rho_log_bin),bounds=my_bound)
    con, beta, gamma = popt[0], popt[1], popt[2]
    perr = np.sqrt(np.diag(pcov))
    sigma_c, sigma_b, sigma_g = perr[0], perr[1], perr[2]
    print('c,beta,gamma =',popt)
    print('err =',perr)
    return(con, beta, gamma, sigma_c, sigma_b, sigma_g)

def fit_with_NFW_profile(radius,res=0.01,N_bin=30):
    # This function has been tested on semi-analytical (Monte-Carlo like) halos
    # with different sphericities, concentrations, slopes and total particle numbers.
    # The typical error is of ~ 30 %, 10 %, 3% for 10**3, 10**4 and 10**5 particles respectively.
    # Typically, the error is slightly less important for the concentration than for the slopes
    r_log_bin, rho_log_bin, dn, dr = profile_log_r_bin_hist(radius,r_min=res,N_bin=N_bin)
    N_vir_sph = len(radius)
    def profile_NFW_0(r_data,concentration):
        # profile abc: rho(r) = rho_s/(x * (1 + x)**2), x=r/r_s
        # r_data should be in Mpc/h
        # rho_s should be in rho_crit
        # concentration, alpha, beta, gamma are dimensionless
        mass = N_vir_sph # Number of particles in the halo
        Rvir = 1 # should be in the same dimension than the data, so I should let Rvir=1 and put the data (r_data) in Rvir unit
        r_s = Rvir/concentration
        #r_s = r_minus_2 * ((beta - 2)/(2 - gamma))**(1/alpha) # the scale radius in function of r_minus_2 in a alpha, beta, gamma profile
        prof_abc_x_2 = lambda x: (x**(-1+2) * (1 + x)**(-2)) # NFW profile times r**2
        my_int = integrate.quad(lambda x: prof_abc_x_2(x),0,Rvir/r_s)
        rho_s = (mass/(4*np.pi*(r_s**3)*my_int[0]))
        x_data = r_data/r_s # dimensionless
        rho = rho_s * ( x_data**(-1) ) * (1 + x_data)**(-2) # in rho_crit
        return(np.log10(rho))
    ####################################################################################
    # No bounds => Levenberg-Marquardt Algorithm: 
    # Implementation and Theory,” Numerical Analysis, ed. G. A. Watson, Lecture Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
    popt, pcov = curve_fit(profile_NFW_0,r_log_bin,np.log10(rho_log_bin),method='lm')
    con = popt[0]
    perr = np.sqrt(np.diag(pcov))
    sigma_c = perr[0]
    print('curve fit: c =',popt)
    print('err =',perr)
    return(con, sigma_c)

def fit_with_NFW_profile_2(r_data,res=0.01,N_bin=30) :
    print('start')
    # This function has been tested on semi-analytical (Monte-Carlo like) halos
    # with different sphericities, concentrations, slopes and total particle numbers.
    # The typical error is of ~ 30 %, 10 %, 3% for 10**3, 10**4 and 10**5 particles respectively.
    # Typically, the error is slightly less important for the concentration than for the slopes
    #y_data = np.log10(rho_log_bin)
    # define residual function
    r_log_bin, rho_log_bin, dn, dr = profile_log_r_bin_hist(r_data,r_min=res,N_bin=N_bin)
    mass = len(r_data) # Number of particles in the halo
    def profile_NFW_2(radius,conc) :
        # profile NFW: rho(r) = rho_s/(x * (1 + x)**2), x=radius/r_s
        # radius should be in Rvir unit, rho_s should be in rho_crit, concentration, is dimensionless
        Rvir = 1 # should be in the same dimension than the data, so I should let Rvir=1 and put the data (r_data) in Rvir unit
        r_s = Rvir/conc[0]
        A_c = np.log(1 + conc[0]) - conc[0]/(1 + conc[0]) # NFW factor
        rho_s = (mass/(4*np.pi*(r_s**3)*A_c)) # in rho_crit
        x_r = radius/r_s # dimensionless radius
        rho = rho_s * ( x_r**(-1) ) * (1 + x_r)**(-2) # in rho_crit
        return(rho)
    def residual_2(param, x_data, y_data):
        y_th = profile_NFW_2(x_data,param)
        res = np.log10(y_data) - np.log10(y_th)
        # the squared of res is minimized if loss='linear' (default)
        return(res)
    # starting point for the parameters
    p0 = np.array([1])  
    # the squared of res is minimized if loss='linear' (default)
    # otherwise, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
    lsq = least_squares(residual_2, p0, method='lm', args=(r_log_bin,rho_log_bin))
    print(lsq)
    print(lsq.x)
    return()

def profile_NFW(radius,conc,mass) :
    # profile NFW: rho(r) = rho_s/(x * (1 + x)**2), x=radius/r_s
    # it has 2 parameters the mass (in particle mass) and the concentration (dimensionless)
    # radius should be in Rvir unit, rho_s should be in rho_crit
    Rvir = 1 # should be in the same dimension than the data, so I should let Rvir=1 and put the data (r_data) in Rvir unit
    r_s = Rvir/conc
    A_c = np.log(1 + conc) - conc/(1 + conc) # NFW factor
    rho_s = (mass/(4*np.pi*(r_s**3)*A_c)) # in rho_crit
    x_r = radius/r_s # dimensionless radius
    rho = rho_s * ( x_r**(-1) ) * (1 + x_r)**(-2) # in rho_crit
    return(rho)

def NFW_func(x) :
    y = np.log(1+x) - x/(1+x)
    return(y)

def mass_NFW(radius,conc,mass) :
    # masses of a halo of mass mass and concentration conc, inside spheres of radius radius, according to NFW
    # see equation 2 of Bhattacharya+13: https://iopscience.iop.org/article/10.1088/0004-637X/766/1/32/pdf
    # radius should be in Rvir unit
    Rvir = 1
    M_in_sphere_r = NFW_func(conc * radius/Rvir) * mass/NFW_func(conc)
    M_shell_r = np.zeros((len(radius)))
    M_shell_r[1:] = M_in_sphere_r[1:] - M_in_sphere_r[:-1]
    M_shell_r[0] = M_in_sphere_r[0]
    return(M_in_sphere_r,M_shell_r)

def residual_NFW_log(param, x_data, y_data, mass):
    y_th = profile_NFW(x_data, param[0], mass)
    res = np.log10(y_data) - np.log10(y_th)
    # the squared of res is minimized if loss='linear' (default)
    return(res)   

def residual_NFW_Child(param, x_data, mass, dn, dr):
    # it computes the residual for the fit, see the equation 5 of Child+18
    # https://ui.adsabs.harvard.edu/abs/2018ApJ...859...55C/abstract
    rho_NFW = profile_NFW(x_data, param[0], mass)
    dn_dr_NFW = 4 * np.pi * x_data**2 * rho_NFW
    res = np.sqrt(((dn/dr - dn_dr_NFW)**2)/(dn/dr**2)) 
    # the squared of res is minimized if the option loss='linear' (default) in least_squares
    return(res)

def profile_Einasto(radius, alpha, conc, mass) :
    # Einasto profile: rho(r) = rho_minus_2 * exp(-(2/alpha) * ( (radius/r_minus_2)**alpha - 1))
    Rvir = 1
    r_minus_2 = Rvir/conc
    prof_Einasto_x_2 = lambda x: x**2 * np.exp((-2/alpha) * (x**alpha - 1))
    my_int = integrate.quad(lambda x: prof_Einasto_x_2(x),0,Rvir/r_minus_2)
    rho_minus_2 = mass/(4 * np.pi * (r_minus_2**3) * my_int[0])
    rho = rho_minus_2 * np.exp( (-2/alpha) * ( (radius/r_minus_2)**alpha - 1))
    return(rho)

def profile_abg(radius, concentration, alpha, beta, gamma, mass):
    # profile abc: rho(r) = rho_s * ( x**(-gamma) ) * (1 + x**alpha)**((gamma-beta)/alpha), x=r/r_s
    # r_data should be in Mpc/h
    # rho_s should be in rho_crit
    # concentration, alpha, beta, gamma are dimensionless
    Rvir = 1 # should be in the same dimension than the data, so I should let Rvir=1 and put the data (r_data) in Rvir unit
    r_minus_2 = Rvir/concentration
    # in the range: 0 <= gamma < 2 < beta, the following equation is correct
    r_s = r_minus_2 * ((beta - 2)/(2 - gamma))**(1/alpha) # the scale radius in function of r_minus_2 in a alpha, beta, gamma profile
    prof_abc_x_2 = lambda x: (x**(2-gamma)) * (1 + x**alpha)**((gamma - beta)/alpha)
    my_int = integrate.quad(lambda x: prof_abc_x_2(x),0,Rvir/r_s)
    rho_s = (mass/(4*np.pi*(r_s**3)*my_int[0]))
    x_data = radius/r_s # dimensionless
    rho = rho_s * ( x_data**(-gamma) ) * (1 + x_data**alpha)**((gamma-beta)/alpha) # in rho_crit
    return(rho)
    
def residual_Einasto_Child(param, x_data, mass, dn, dr):
    # it computes the residual for the fit, see the equation 5 of Child+18
    # https://ui.adsabs.harvard.edu/abs/2018ApJ...859...55C/abstract
    rho_Einasto = profile_Einasto(x_data, param[0], param[1], mass)
    dn_dr_Einasto = 4 * np.pi * x_data**2 * rho_Einasto
    res = np.sqrt(((dn/dr - dn_dr_Einasto)**2)/(dn/dr**2)) 
    # the squared of res is minimized if the option loss='linear' (default) in least_squares
    return(res)

def residual_abg_Child(param, x_data, mass, dn, dr):
    # it computes the residual for the fit, see the equation 5 of Child+18 but for alpha beta gamma profile
    # https://ui.adsabs.harvard.edu/abs/2018ApJ...859...55C/abstract
    conc, alpha, beta, gamma = param[0], param[1], param[2], param[3]
    rho_abg = profile_abg(x_data, conc, alpha, beta, gamma, mass)
    dn_dr_abg = 4 * np.pi * x_data**2 * rho_abg
    res = np.sqrt(((dn/dr - dn_dr_abg)**2)/(dn/dr**2)) 
    # the squared of res is minimized if the option loss='linear' (default) in least_squares
    return(res)

def goodness_of_fit(rho_data,rho_th,constraint=0) :
    chi_2 = np.sum(((rho_data-rho_th)**2)/rho_data**2)
    chi_2_norm = chi_2/(len(rho_data) -constraint)
    return(chi_2,chi_2_norm)

def residual_NFW_Bhattacharya(param, x_data, mass, dn):
    M_in_sphere_r, M_shell_r = mass_NFW(x_data, param[0], mass)
    res = ((dn - M_shell_r)**2)/dn
    return(res)
    
def do_plot() :
    plt.rc('font', family='serif',size=16)
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
    fig, ax = plt.subplots(figsize=[10,10]) # plot
    ax.tick_params(which='both', width=0.5, direction="in",
                   labeltop=False,labelbottom=True, labelleft=True, labelright=False)
    x = range(1,N_samples+1)
    y = np.ones((N_samples))
    plt.scatter(x,err_log,c='r')
    plt.plot(x,np.mean(err_log) * y,ls='-',c='r',label='% error mean, log')
    #plt.fill_between(x, (np.mean(err_log)+np.std(err_log)) * y, (np.mean(err_log)-np.std(err_log)) * y,color='r',alpha=0.2)
    plt.hlines(-np.mean(err_log_cov)/conc_true,1,N_samples,ls='--',colors='r',label='$-\sigma_{cov}/c_{true}$, log')
    
    plt.scatter(x,err_Child,c='b',)
    plt.plot(x,np.mean(err_Child) * y,ls='-',c='b',label='% error mean, Poisson weight')
    #plt.fill_between(x, (np.mean(err_Child)+np.std(err_Child)) * y, (np.mean(err_Child)-np.std(err_Child)) * y,color='b',alpha=0.2)
    plt.hlines(-np.mean(err_Child_cov)/conc_true,1,N_samples,ls='--',colors='b', label='$-\sigma_{cov}/c_{true}$, Poisson weight')
    plt.ylabel('$\dfrac{c_{fit} - c_{true}}{c_{true}}$',labelpad=-5)
    plt.legend(loc='best', ncol=1, frameon=False)
    plt.title('Fit error on $N_{vir} = 10^'+str(np.int(np.log10(N_part)))+'$ particles Monte-Carlo haloes')
    path_and_name = path+'fit_error_on_Monte_Carlo_halos_with_'+str(N_part)+'_particles.pdf'
    #plt.savefig(path_and_name,format='pdf',transparent=True)
    plt.show()
    return()

if __name__ == '__main__':
    path = '../../../../DEUSS_648Mpc_2048_particles/output_not_MF/cosmo_DE_z_impact/test/SA_halos/'
    N_samples = 1
    N_part = 100000
    res = 0.01
    N_bin = 20
    err_log, err_Child = np.zeros((N_samples)), np.zeros((N_samples))
    err_Einasto, err_Einasto_cov = np.zeros((N_samples)), np.zeros((N_samples))
    err_abg, err_abg_cov = np.zeros((N_samples)), np.zeros((N_samples))
    err_log_cov, err_Child_cov = np.zeros((N_samples)), np.zeros((N_samples))
    for s in range(N_samples) :
        pos = np.loadtxt(path+'halo_Npart_'+str(N_part)+'_abg_131_c_5_abc_111_'+str(s)+'.dat')
        conc_true = 5
        r_data = np.sqrt(pos[:,0]**2 + (pos[:,1])**2 + (pos[:,2])**2)
        ind = np.where(r_data < 1)[0]
        #con, beta, gamma, sigma_c, sigma_b, sigma_g = fit_with_abg_profile(radius[ind],res=res)
        ############################################################################# 
        # curve_fit
        #con, sigma_c = fit_with_NFW_profile(r_data,res=res)
        #fit_with_NFW_profile_2(r_data,res=res)
        ######################################################################################
        # lsq with log residual
        r_log_bin, rho_log_bin, dn, dr = profile_log_r_bin_hist(r_data,r_min=res,N_bin=N_bin)
        mass = len(r_data) # Number of particles in the halo
        p0 = np.array([1])  
        # the squared of res is minimized if loss='linear' (default)
        # otherwise, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
        lsq = least_squares(residual_NFW_log, p0, method='lm', args=(r_log_bin, rho_log_bin, mass))
        #print(lsq)
        conc_lsq_log = lsq.x
        print('lsq, log(dens) c =',conc_lsq_log)
        err_log[s] = (conc_lsq_log - conc_true)/conc_true
        Jac = lsq.jac
        Jac_t = Jac.T
        hess_inv = np.linalg.inv(np.dot(Jac_t,Jac))
        res_var = (residual_NFW_log(conc_lsq_log, r_log_bin, rho_log_bin, mass)**2).sum()/(len(rho_log_bin)-len(p0))
        err_log_cov[s] = np.sqrt(np.diag(hess_inv*res_var))
        print('lsq, log(dens) err=',err_log_cov[s])
        dn = dn[1:]
        dr_save = dr
        dr = dr[1:]
        y_data = dn/dr
        chi_2_log = goodness_of_fit(np.log10(y_data),np.log10(profile_NFW(r_log_bin,conc_lsq_log,mass)))
        print('chi_2_log =',chi_2_log)
        # the squared of res is minimized if loss='linear' (default)
        # otherwise, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
        ############################################################################################
        # lsq à la Child+18: https://ui.adsabs.harvard.edu/abs/2018ApJ...859...55C/abstract
        lsq = least_squares(residual_NFW_Child, p0, method='lm', args=(r_log_bin, mass, dn, dr))
        conc_Child = lsq.x
        print('lsq, à la Child+18 c =',conc_Child)
        err_Child[s] = (conc_Child - conc_true)/conc_true
        # https://math.stackexchange.com/questions/2349026/why-is-the-approximation-of-hessian-jtj-reasonable
        Jac = lsq.jac
        Jac_t = Jac.T
        hess_inv = np.linalg.inv(np.dot(Jac_t,Jac))
        res_var = (residual_NFW_Child(conc_Child,r_log_bin, mass, dn, dr)**2).sum()/(len(dn)-len(p0))
        err_Child_cov[s] = np.sqrt(np.diag(hess_inv*res_var))
        print('lsq, à la Child+18 err=',err_Child_cov[s])
        chi_2_Child = goodness_of_fit(np.log10(y_data),np.log10(profile_NFW(r_log_bin,conc_Child[0],mass)))
        print('chi_2 Child =',chi_2_Child)
        
        # lsq à la Child+18: https://ui.adsabs.harvard.edu/abs/2018ApJ...859...55C/abstract
        # but Einasto profiles
        p0_Einasto = np.array([0.2,1])
        lsq = least_squares(residual_Einasto_Child, p0_Einasto, bounds=([0.05,1],[0.6,100]), method='trf', args=(r_log_bin, mass, dn, dr))
        p_Einasto = lsq.x
        print('lsq, à la Child+18 for Einasto alpha and c =',p_Einasto)
        err_Einasto[s] = (p_Einasto[1] - conc_true)/conc_true
        # https://math.stackexchange.com/questions/2349026/why-is-the-approximation-of-hessian-jtj-reasonable
        Jac = lsq.jac
        Jac_t = Jac.T
        hess_inv = np.linalg.inv(np.dot(Jac_t,Jac))
        res_var = (residual_Einasto_Child(p_Einasto,r_log_bin, mass, dn, dr)**2).sum()/(len(dn)-len(p0_Einasto))
        err_Einasto_cov[s] = np.sqrt(np.diag(hess_inv*res_var))[1]
        print('lsq, à la Child+18 for Einasto err=',np.sqrt(np.diag(hess_inv*res_var)))
        chi_2_Einasto = goodness_of_fit(np.log10(y_data),np.log10(profile_Einasto(r_log_bin,p_Einasto[0],p_Einasto[1],mass)))
        print('chi_2 Einasto =',chi_2_Einasto)
        
        # lsq à la Child+18: https://ui.adsabs.harvard.edu/abs/2018ApJ...859...55C/abstract
        # but abg profiles
        p0_abg = np.array([1,1,3,1])
        lsq = least_squares(residual_abg_Child, p0_abg, bounds=([1,0.5,2.1,0],[100,2,5,1.9]), method='trf', args=(r_log_bin, mass, dn, dr))
        p_abg = lsq.x
        print('lsq, à la Child+18 for abg c, a, b, g =',p_abg)
        err_abg[s] = (p_abg[0] - conc_true)/conc_true
        # https://math.stackexchange.com/questions/2349026/why-is-the-approximation-of-hessian-jtj-reasonable
        Jac = lsq.jac
        Jac_t = Jac.T
        hess_inv = np.linalg.inv(np.dot(Jac_t,Jac))
        res_var = (residual_abg_Child(p_abg,r_log_bin, mass, dn, dr)**2).sum()/(len(dn)-len(p0_abg))
        err_abg_cov[s] = np.sqrt(np.diag(hess_inv*res_var))[1]
        print('lsq, à la Child+18 for abg err=',np.sqrt(np.diag(hess_inv*res_var)))
        chi_2_abg = goodness_of_fit(np.log10(y_data),np.log10(profile_abg(r_log_bin,p_abg[0],p_abg[1],p_abg[2],p_abg[3],mass)))
        print('chi_2 abg =',chi_2_abg)
        
        ###########################################################################################
        # lsq à la Bhattacharya+13: https://ui.adsabs.harvard.edu/abs/2013ApJ...766...32B/abstract
        '''r_log_shell = np.cumsum(dr_save)
        lsq = least_squares(residual_NFW_Bhattacharya, p0, method='lm', args=(r_log_shell[1:], mass, dn))
        conc_Bat = lsq.x
        #print('lsq, à la Bhattacharya+13 c =',conc_Bat)
        Jac = lsq.jac
        Jac_t = Jac.T
        hess_inv = np.linalg.inv(np.dot(Jac_t,Jac))
        res_var = (residual_NFW_Bhattacharya(conc_Child,r_log_shell[1:], mass, dn)**2).sum()/(len(dn)-len(p0))
        #print('lsq, à la Bhattacharya+13 err=',np.sqrt(np.diag(hess_inv*res_var)))
        '''
    #do_plot()
    #print('end')
    #print('err cov log =',np.mean(err_log_cov)/conc_true)
    #print('err std log',np.std(err_log))
    #print('err cov Child =',np.mean(err_Child_cov)/conc_true)
    #print('err std Child',np.std(err_Child))