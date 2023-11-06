# ASMC-SURR-HF
Adaptive Sequential Monte Carlo python codes combined with surrogate solvers, for posterior inference and evidence computation.

This codes correspond to the article by Amaya et al. (Multifidelity adaptive sequential Monte Carlo applied to geophysical inversion. Submitted to Geophysical Journal International). It is a Python 3.7 implementation of the Adaptive Sequential Monte Carlo (ASMC) method (Zhou et al., 2016; algorithm 4) to estimate 
the posterior probability density function (PDF) and the evidence (marginal likelihood) trough a Bayesian inversion. ASMC is a particle approach that relies on importance sampling over a sequence 
of intermediate distributions (power posteriors) that link the prior and the posterior PDF. Each power posterior is approximated by updating the particle importance weights and states using a small 
pre-defined number MCMC proposal steps. ASMC method adaptively tunes the sequence of power posteriors and performs resampling of particles when the variance of their importance weights 
becomes too large.

This particular implementation relies on polinomial chaoes expansion (PCE) surrogates, although other types of surrogate solver can be used. 
The PCE surrogates are trained using Uqlab https://www.uqlab.com/ . 

## Test cases
The test case is a synthetic ground penetrating radar tomography modified from Meles et al. (2022).

## Codes 
run_asmc_surr.py : control the user-defined parameters and run the inversion. 

asmc_surr.py : main asmc code.

asmc_func_surr.py : contain the auxiliar functions called by asmc.py.

##Note that there is a file missing called PCADATA.m (too big for github), which contains the coefficients learnt for the model principal component decomposition of this example. 
In case you would like to reproduce the exact teste case, I can easily provide you with the file. 

## Performing the inversion
Modify the user-defined parameters and run run_asmc_surr.py. 

## Citation :
Amaya, M., Linde, N., & Laloy, E. (2023). Multifidelity adaptive sequential Monte Carlo applied to geophysical inversion. Submitted to Geophysical Journal International.


## References:
Amaya, M., Linde, N., Laloy, E. (2023). Multifidelity adaptive sequential Monte Carlo applied to geophysical inversion. Submitted to Geophysical Journal International. 

Amaya, M., Linde, N., & Laloy, E. (2021). Adaptive sequential Monte Carlo for posterior inference and model selection among complex geological priors. Geophysical Journal International, 226(2), 1220-1238.

Laloy, E., & Vrugt, J. A. (2012). High‐dimensional posterior exploration of hydrologic models using multiple‐try DREAM (ZS) and high‐performance computing. Water Resources Research, 48(1).

Marelli, S. & Sudret, B., 2014. UQLab: A framework for uncertainty quantification in Matlab. In: Beer, M., Au, S. and Hall, J.W., Eds., Vulnerability, Uncertainty, and Risk: Quantification, Mitigation, and Management, pp. 2554–2563.

Marelli, S., Lu¨then, N., & Sudret, B., 2022. UQLab user manual – Polynomial chaos expansions, Tech. rep., Chair of Risk, Safety and Uncertainty Quantification, ETH Zurich, Switzerland, Report UQLab-V2.0-104.

Meles, G. A., N. Linde, and S. Marelli (2022), Bayesian tomography with prior-knowledgebased parametrization and surrogatemodelling, Geophysical Journal International, 231(1),673–691.

Ter Braak, C. J., & Vrugt, J. A. (2008). Differential evolution Markov chain with snooker updater and fewer chains. Statistics and Computing, 18(4), 435-446.

Vrugt, J. A., ter Braak, C., Diks, C., Robinson, B. A., Hyman, J. M., & Higdon, D. (2009). Accelerating Markov chain Monte Carlo simulation by differential evolution with self-adaptive randomized subspace sampling. International Journal of Nonlin ear Sciences and Numerical Simu- lation, 10(3), 273–290.

Zhou, Y., Johansen, A. M., & Aston, J. A., (2016). Toward automatic model comparison: an adaptive sequential Monte Carlo approach, Journal of Computational and Graphical Statistics,69925(3), 701–726.



## License
See LICENSE.txt


## Contact
Macarena Amaya (macarena.amaya@unil.ch)
