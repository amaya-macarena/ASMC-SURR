# -*- coding: utf-8 -*-
"""
Adaptive Sequential Monte Carlo inversion combined with surrogate model updating, for posterior inference and evidence computation.  
This code controls the user-defined parameters and calls the main program asmc_surr.py to performe the inversion and saves the results. 

This codes correspond to the article by Amaya et al. (2022). It is a Python 3.7 implementation of the Adaptive Sequential Monte Carlo (ASMC) method (Zhou et al., 2016; algorithm 4) to estimate 
the posterior probability density function (PDF) and the evidence (marginal likelihood) trough a Bayesian inversion. ASMC is a particle approach that relies on importance sampling over a sequence 
of intermediate distributions (power posteriors) that link the prior and the posterior PDF. Each power posterior is approximated by updating the particle importance weights and states using a small 
pre-defined number MCMC proposal steps. ASMC method adaptively tunes the sequence of power posteriors and performs resampling of particles when the variance of their importance weights 
becomes too large.

The test case is a synthetic ground penetrating radar tomography from Meles et al. (2022).

This ASMC implementation (referred to as ASMC-SURR-HF, Algorithm 1 in Amaya et al. (2023)) includes:

- an adaptation of the DREAMzs algorithm to perform MCMC steps approximating each power posterior (ter Braak and Vrugt, 2008; Vrugt, 2009; Laloy and Vrugt, 2012)

- the 'high-fidelity' solver is the time2d algorithm by Podvin & Lecomte (1991) that computes the cross-hole ground penetrating radar (GPR) tomography first-arrival times.

- the 'low-fidelity' solvers are polinomial chaos expansion surrogates obtained using the Matlab-based package UQLab (Marelli & Sudret, 2014);
details on the implementation of PCE can be found in Marelli et al. (2022).

In case you have a question or if you find a bug, please write me an email to macarena.amaya@unil.ch. 


References:

Amaya, M., Linde, N., Laloy, E. (2023). Multifidelity adaptive sequential Monte Carlo applied to geophysical inversion. Submitted to Geophysical Journal International. 

Amaya, M., Linde, N., & Laloy, E. (2021). Adaptive sequential Monte Carlo for posterior inference and model selection among complex geological priors. Geophysical Journal International, 226(2), 1220-1238.

Laloy, E., & Vrugt, J. A. (2012). High‐dimensional posterior exploration of hydrologic models using multiple‐try DREAM (ZS) and high‐performance computing. Water Resources Research, 48(1).

Marelli, S. & Sudret, B., 2014. UQLab: A framework for uncertainty quantification in Matlab. In: Beer, M., Au, S. and Hall, J.W., Eds., Vulnerability, Uncertainty, and Risk: Quantification, Mitigation, and Management, pp. 2554–2563.

Marelli, S., Lu¨then, N., & Sudret, B., 2022. UQLab user manual – Polynomial chaos expansions, Tech. rep., Chair of Risk, Safety and Uncertainty Quantification, ETH Zurich, Switzerland, Report UQLab-V2.0-104.

Meles, G. A., N. Linde, and S. Marelli (2022), Bayesian tomography with prior-knowledgebased parametrization and surrogatemodelling, Geophysical Journal International, 231(1),673–691.

Ter Braak, C. J., & Vrugt, J. A. (2008). Differential evolution Markov chain with snooker updater and fewer chains. Statistics and Computing, 18(4), 435-446.

Vrugt, J. A., ter Braak, C., Diks, C., Robinson, B. A., Hyman, J. M., & Higdon, D. (2009). Accelerating Markov chain Monte Carlo simulation by differential evolution with self-adaptive randomized subspace sampling. International Journal of Nonlin ear Sciences and Numerical Simu- lation, 10(3), 273–290.

Zhou, Y., Johansen, A. M., & Aston, J. A., (2016). Toward automatic model comparison: an adaptive sequential Monte Carlo approach, Journal of Computational and Graphical Statistics,69925(3), 701–726.

                                                                                                                                                                                                               
"""
import os
import time
import numpy as np
import shutil

work_dir=os.getcwd()

import asmc_surr

CaseStudy=1  
Restart=False
    

if  CaseStudy==1: 

    # User defined parameters:
    seq=50 # Number of particles (N)
    thin=500# Thinning parameter for saving the sampled likelihoods
    steps=500# Iterations per intermediate distribution (K)
    CESSf_div=0.99# targeted CESS (CESS_op) 
    ESSf_div=0.3 # ESS treshold (ESS*) 
    AR_max=35.0# Max acceptance rate before decreasing proposal scale
    AR_min=15.0 # Min acceptance rate before decreasing proposal scale
    
#    Init='True'
    
    ndraw=seq*500000# Set a high number of iterations to stop in case the beta sequence becomes too long due to a bad choice of CESS  
    tune_phi=seq*500000 # Iterations in which the proposal scale is tune it (for our case we tune it all along the run, so we define a high number of iterations that is not reached). 

    
    #Decide if to run forward solver in parallel
    
        
    DoParallel=False
    parallel_jobs=seq
    MakeNewDir=True
    
    if MakeNewDir==False:
        src_dir=work_dir+'/forward_setup_0'
        for i in range(1,parallel_jobs+1):
            dst_dir=work_dir+'/forward_setup_'+str(i)
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir,dst_dir)
            
     
#% Run the ASMC-SGR algorithm
            
if __name__ == '__main__':
    
    #start_time = time.time()

    q=asmc_surr.Sampler(main_dir=work_dir,CaseStudy=CaseStudy,seq=seq,ndraw=ndraw,parallel_jobs=seq,steps=steps,
                   parallelUpdate = 1,pCR=False,thin=thin,nCR=3,DEpairs=1,pJumpRate_one=0.2,BoundHandling='Fold',
                   lik_sigma_est=False,DoParallel=DoParallel,CESSf_div=CESSf_div,ESSf_div=ESSf_div,AR_min=AR_min,AR_max=AR_max,tune_phi=tune_phi)
    
    print("Iterating")
    
    if Restart:
        tmpFilePath=work_dir+'/out_tmp.pkl'
    else:
        tmpFilePath=None 
    
#    Sequences, Z, OutDiag, fx, MCMCPar, MCMCVar, beta_run, jr_seq, weig_seq, CESS_ev, increment, ESS_ev, evid_cont, evid_ev, weights_unnorm, new_weight_ev, weig_cont, eve_seq = q.sample(RestartFilePath=tmpFilePath)
    Sequences, OutDiag, MCMCPar, MCMCVar, beta_run, phi_used_ev, phi_ev, weig_seq, CESS_ev, increment, ESS_ev, evid_cont, evid_ev, weights_unnorm, new_weight_ev, weig_cont, eve_seq, last_models,last_likelihoods,intermediate_models,beta_run_surr,beta_K_steps,alpha_corrections = q.sample(RestartFilePath=tmpFilePath)
    
    os.mkdir(work_dir+'/results')
    dir_res=work_dir+'/results'
    np.save(work_dir+'/results/Sequences_states.npy',Sequences) # Evolution of the states for every particle (latent parameters) and its likelihood
    np.save(work_dir+'/results/AR.npy',OutDiag.AR) # Acceptance Rate
    np.save(work_dir+'/results/beta_seq.npy',beta_run) # Sequence that defines the intermediate distributions (resulting from the adaptive procedure) 
    np.save(work_dir+'/results/beta_seq_surr.npy',beta_run_surr)    
    np.save(work_dir+'/results/beta_increment.npy',increment)
    np.save(work_dir+'/results/beta_K_steps.npy',beta_K_steps)       
    np.save(work_dir+'/results/alpha_corrections.npy',alpha_corrections)             
    np.save(work_dir+'/results/CESS.npy',CESS_ev)
    np.save(work_dir+'/results/ESS.npy',ESS_ev)    
    np.save(work_dir+'/results/phi_used_ev.npy',phi_used_ev)
    np.save(work_dir+'/results/phi_ev.npy',phi_ev)# Proposal scale evolution
    np.save(work_dir+'/results/weig_ev.npy',weig_seq) # Weights evolution	
    np.save(work_dir+'/results/weig_cont.npy',weig_cont) # Weights evolution	 	
    np.save(work_dir+'/results/evid_ev.npy',evid_ev) # Evidence evolution
    np.save(work_dir+'/results/eve_ev.npy',eve_seq) # Eve index evolution
    np.save(work_dir+'/results/last_models.npy',last_models)
    np.save(work_dir+'/results/last_likelihoods.npy',last_likelihoods)
    np.save(work_dir+'/results/intermediate_models.npy ',intermediate_models )      