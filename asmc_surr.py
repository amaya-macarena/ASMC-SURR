"""
Adaptive Sequential Monte Carlo inversion combined with surrogate model updating, for posterior inference and evidence computation.  
This is the main code for the inversion, to modify the user-defined parameters and run the inversion please see run_asmc_surr.py

This codes correspond to the article by Amaya et al. (2023). It is a Python 3.7 implementation of the Adaptive Sequential Monte Carlo (ASMC) method (Zhou et al., 2016; algorithm 4) to estimate 
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
from __future__ import print_function

import numpy as np
import numpy.matlib as matlib
try:
    import cPickle as pickle
except:
    import pickle

import time

from statsmodels.stats.weightstats import DescrStatsW

from scipy.stats import triang

from asmc_surr_func import* # This imports all ASMC-SURR and inverse problem-related functions

import sys

from attrdict import AttrDict

import scipy.io

import matlab.engine

MCMCPar=AttrDict()

MCMCVar=AttrDict()

Measurement=AttrDict()

OutDiag=AttrDict()

Extra=AttrDict()

import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed




class Sampler:

    

    
    def __init__(self, main_dir=None,CaseStudy=0,seq = 3,ndraw=10000,thin = 1,  nCR = 3, 
                 DEpairs = 3, parallelUpdate = 1.0, pCR=True,k=10,pJumpRate_one=0.2,
                 steps=100,savemodout=False, saveout=True,save_tmp_out=True,Prior='LHS',
                 DoParallel=True,eps=5e-2,BoundHandling='Reflect',
                 lik_sigma_est=False,parallel_jobs=4,rng_seed=123,it_b=10,jr_factor=0.2,CESSf_div=0.999993,ESSf_div=0.5,AR_min=15.0,AR_max=30.0,tune_phi=1000):
        
        
        
        
        self.CaseStudy=CaseStudy
        MCMCPar.seq = seq
        MCMCPar.ndraw=ndraw
        MCMCPar.thin=thin
        MCMCPar.nCR=nCR
        MCMCPar.DEpairs=DEpairs
        MCMCPar.parallelUpdate=parallelUpdate
        MCMCPar.Do_pCR=pCR
        MCMCPar.k=k
        MCMCPar.pJumpRate_one=pJumpRate_one
        MCMCPar.steps=steps
        MCMCPar.savemodout=savemodout
        MCMCPar.saveout=saveout  
        MCMCPar.save_tmp_out=save_tmp_out  
        MCMCPar.Prior=Prior
        MCMCPar.DoParallel=DoParallel
        MCMCPar.eps = eps
        MCMCPar.BoundHandling = BoundHandling
        MCMCPar.lik_sigma_est=lik_sigma_est
        Extra.n_jobs=parallel_jobs
        Extra.main_dir=main_dir
        np.random.seed(seed=None) 
        MCMCPar.it_b=it_b
        MCMCPar.jr_factor=jr_factor
        MCMCPar.AR_min=AR_min
        MCMCPar.AR_max=AR_max      
        MCMCPar.ESSf_div=ESSf_div
        MCMCPar.CESSf_div=CESSf_div
        MCMCPar.tune_phi=tune_phi



        if self.CaseStudy==1:   
            
### Load data and data covariance matrix   

            Measurement.MeasData=np.load('Data_contaminated_eikonal_new_0.5ns_69tt.npy')

            mat2 = scipy.io.loadmat('corr3_SEIK_69tt_COV_Loo_PCE200_100.mat')
            an_array2 = np.array(list(mat2.items()))
            mod_err_cov=an_array2[4,1]
            total_cov = an_array2[5,1]
            Measurement.mod_err_cov=mod_err_cov #* 1e18
            Measurement.Sigma = 0.5 #1e-9np.size(Measurement.MeasData) 
            mat3 = scipy.io.loadmat('PCA_COV_100_3000tset_69tt.mat')
            an_array3 = np.array(list(mat3.items()))
            pca = an_array3[4,1]       
            Measurement.cov_pca=pca              
            Measurement.data_cov=np.identity(69)*(0.5**2)               
            
            MCMCPar.lik = 5 #2  
 
            Measurement.total_cov = total_cov + pca
            
            np.save('Covariance_matrix_all_surr.npy',Measurement.total_cov)    
            
            Measurement.N=np.size(Measurement.MeasData) 

### Model size, likelihood type and proposal scale
            
            ModelName='nonlinear_gpr_tomo'
         
            MCMCPar.cond_type=2
            
            if MCMCPar.cond_type==2:

                MCMCPar.phi=np.ones(MCMCPar.seq)*0.1
                       
#           Define the list to choose from in the binary search that looks for the adaptive inverse temperature decrements.
                       
            gamma=np.linspace(-1,1,400)

            MCMCPar.gamma=gamma
                
        self.MCMCPar=MCMCPar
        self.Measurement=Measurement
        self.Extra=Extra
        self.ModelName=ModelName

### Initialize uqlab, calculate prior (marginals) and build covariance matrix for proposal
       
        data_set=300
        npca=100
        ypca=69 # Number of data points
        MCMCPar.npca=npca
        MCMCPar.ypca=ypca
        MCMCPar.data_set=data_set

        # DATA PREPARATION;
        nr_data=data_set # size(X)
        nr_val=100 #size Xval
        nr_fulldata=nr_data+nr_val

        
 ##############################################        

######    Initialization for the main sampling loop      

    def sample(self,RestartFilePath=None):
        
        start_time1b = time.time()
        
## Load the initial surrogate (needs to have been trained in advance)
        
        eng = matlab.engine.start_matlab()
        timeeng=time.time()
        [myPCE,myInput,Moments]=eng.PCE_training_loadsession(MCMCPar.npca,MCMCPar.ypca,MCMCPar.data_set,MCMCPar.seq,nargout=3)
        print('time call matlab engine and train PCE',time.time()-timeeng)

## Build diagional matrix to perturbate models and propose candidates
        
        Moments = np.asarray(Moments)
        MCMCPar.prior_std=np.zeros(MCMCPar.npca)        
        MCMCPar.prior_mean=np.zeros(MCMCPar.npca)
                
        for i in range(MCMCPar.npca):
                    
            MCMCPar.prior_std[i]=Moments[i,1]
            MCMCPar.prior_mean[i]=Moments[i,0]
                    
        MCMCPar.matrix_std=np.diag(MCMCPar.prior_std)       
        np.save('matrix_std.npy',MCMCPar.matrix_std)        
        
        if not(RestartFilePath is None):
            print('This is a restart')
            with open(RestartFilePath, 'rb') as fin:
                tmp_obj = pickle.load(fin)
            self.Sequences=tmp_obj['Sequences']
            self.Z=tmp_obj['Z']
            self.OutDiag=tmp_obj['OutDiag']
            self.fx=tmp_obj['fx']
            self.MCMCPar=tmp_obj['MCMCPar']
            self.MCMCVar=tmp_obj['MCMCVar']
            self.Measurement=tmp_obj['Measurement']
            self.ModelName=tmp_obj['ModelName']
            self.Extra=tmp_obj['Extra']
            del tmp_obj
            
            self.ndim=self.MCMCPar.n
#                
            self.MCMCPar.ndraw = 2 * self.MCMCPar.ndraw
            
            # Reset rng
            np.random.seed(np.floor(time.time()).astype('int'))
            
### Extend Sequences, Z, OutDiag.AR,OutDiag.Rstat and OutDiag.CR
            self.Sequences=np.concatenate((self.Sequences,np.zeros((self.Sequences.shape))),axis=0)
            self.OutDiag.AR=np.concatenate((self.OutDiag.AR,np.zeros((self.OutDiag.AR.shape))),axis=0)
            self.OutDiag.CR=np.concatenate((self.OutDiag.CR,np.zeros((self.OutDiag.CR.shape))),axis=0)
                
        else:

            start_time = time.time()           
            Iter=self.MCMCPar.seq
            iteration=2
            iloc=0
            T=0
            
            self.MCMCPar.n=MCMCPar.npca
            m_currentm=scipy.io.loadmat('prior_init_models.mat') #load the prior models created during PCE_training    
            an_array = np.array(list(m_currentm.items()))
            m_current=an_array[3,1]
            np.save('prior_init_models.npy',m_current)
        
            A=matlab.double(m_current.tolist())
            timeev=time.time()
            [YPCEm,prior_density]=eng.PCE_eval(A,myPCE,myInput,nargout=2)
            timeevf=time.time()
            print('time evaluating PCE',timeevf-timeev)
            YPCE=np.array(YPCEm)
            logprior_density=np.array(prior_density)
            np.save('prior_density.npy',prior_density)
            logprior_density=np.array(prior_density)

            
            self.MCMCPar.CR=np.cumsum((1.0/self.MCMCPar.nCR)*np.ones((1,self.MCMCPar.nCR)))
            Nelem=np.floor(self.MCMCPar.ndraw/self.MCMCPar.seq)++self.MCMCPar.seq*2
            OutDiag.CR=np.zeros((np.int(np.floor(Nelem/self.MCMCPar.steps))+2,self.MCMCPar.nCR+1))
            OutDiag.AR=np.zeros((np.int(np.floor(Nelem/self.MCMCPar.steps))+2,2))
            OutDiag.AR[0,:] = np.array([self.MCMCPar.seq,-1])
            pCR = (1.0/self.MCMCPar.nCR) * np.ones((1,self.MCMCPar.nCR))
            
            # Calculate the actual CR values based on pCR
            CR,lCR = GenCR(self.MCMCPar,pCR)  
            
            if self.MCMCPar.savemodout:
                self.fx = np.zeros((self.Measurement.N,np.int(np.floor(self.MCMCPar.ndraw/self.MCMCPar.thin))))
                MCMCVar.m_func = self.MCMCPar.seq     

            self.Sequences = np.empty((np.int(np.floor(Nelem/self.MCMCPar.thin))+1,self.MCMCPar.n+3,self.MCMCPar.seq))
               
            self.MCMCPar.Table_JumpRate=np.zeros((self.MCMCPar.n,self.MCMCPar.DEpairs))
            for zz in range(0,self.MCMCPar.DEpairs):
                self.MCMCPar.Table_JumpRate[:,zz] = 2.38/np.sqrt(2 * (zz+1) * np.linspace(1,self.MCMCPar.n,self.MCMCPar.n).T)


            of,log_p = CompLikelihood(m_current,YPCE,self.MCMCPar,self.Measurement,self.Extra,self.Measurement.total_cov)#self.Measurement.total_cov)

            print('m_current.shape',m_current.shape)
            print('of.shape',of.shape)
            print('log_p.shape',log_p.shape)            
            print('logprior_density',logprior_density)                        

            X = np.concatenate((m_current,of,log_p,logprior_density),axis=1)
            
            Xfx = YPCE
        
            
            if self.MCMCPar.savemodout==True:
                self.fx=fx0
            else:
                self.fx=None


            aux_in=np.reshape(X.T,(1,self.MCMCPar.n+3,self.MCMCPar.seq))
            self.Sequences[0,:,:self.MCMCPar.seq] = aux_in[0,:,:self.MCMCPar.seq]
            # Store N_CR
            OutDiag.CR[0,:MCMCPar.nCR+1] = np.concatenate((np.array([Iter]).reshape((1,1)),pCR),axis=1)
            delta_tot = np.zeros((1,self.MCMCPar.nCR))
        
            self.OutDiag=OutDiag    
            MCMCVar.Iter=Iter
            MCMCVar.iteration=iteration
            MCMCVar.iloc=iloc; MCMCVar.T=T; MCMCVar.X=X
            MCMCVar.Xfx=Xfx; MCMCVar.CR=CR; MCMCVar.pCR=pCR
            MCMCVar.lCR=lCR; MCMCVar.delta_tot=delta_tot
            self.MCMCVar=MCMCVar
            MCMCVar.m_current=m_current
            
            if self.MCMCPar.save_tmp_out==True:
                with open('out_tmp'+'.pkl','wb') as f:
                     pickle.dump({'Sequences':self.Sequences,
                     'OutDiag':self.OutDiag,'fx':self.fx,'MCMCPar':self.MCMCPar,
                     'MCMCVar':self.MCMCVar,'Measurement':self.Measurement,
                     'ModelName':self.ModelName,'Extra':self.Extra},f, protocol=pickle.HIGHEST_PROTOCOL)
        
            end_time = time.time()
                 
            print("init_sampling took %5.4f seconds." % (end_time - start_time))                            

### Initialize variables and arrays to store the results:

        prov_AR=30     # (no real meaning, just to not change the jr_scale on the first loop),   
        beta_run=[]
        beta_run_surr=[]
        beta_K_steps=[]
        increment=[]
        phi_used=np.zeros(self.MCMCPar.seq)
        phi_used_ev=np.zeros((1,self.MCMCPar.seq))
        phi_ev=np.zeros((1,self.MCMCPar.seq))
        
        likelihood_ev=[]
        prior_ev=[]
        weig_seq=[]
        weig_cont=[]
        weights_unnorm=[]
        weig_unn=np.ones(self.MCMCPar.seq)
        norm_weight=np.ones(self.MCMCPar.seq)/self.MCMCPar.seq
        new_weight_ev=[]
        weig_seq=np.append(weig_seq,norm_weight)
        
        CESSf=self.MCMCPar.CESSf_div*self.MCMCPar.seq
        CESS_ev=[]
        ESS_ev=[]
        beta=0.
        beta_full=0.
        omega=-6
        beta_run=np.append(beta_run,beta)
        beta_run_surr=np.append(beta_run_surr,beta)
        beta_K_steps=np.append(beta_K_steps,beta)
        beta_aux=1.
        update_surr=0.
       
        ind=0
        num_inc=self.MCMCPar.gamma.shape[0]  
        evid_cont=[]
        evid_ev=[]
        evid_evolution=0       
        alpha_corrections=[]   
        eve_prev=np.arange(0,self.MCMCPar.seq,dtype=int)
        anc_prev=np.arange(0,self.MCMCPar.seq,dtype=int)       
        eve_seq=[]
        eve_seq=np.append(eve_seq,eve_prev)

        full_physics=False   
        YPCE=np.zeros((MCMCPar.seq,MCMCPar.ypca))        
                  
        PCADATA=scipy.io.loadmat('PCADATA.mat')
        an_array = np.array(list(PCADATA.items()))
        coeff=an_array[4,1]
        MED=an_array[3,1]
        D=coeff[:,1:self.MCMCPar.npca]
        
        ##Surrogate updating parameters
        temp_surr_update=35.
        cov_end_selec=True
        trainig_max_size=2000
        const_training_size=False       
        X_train=[]
        Y_train=[]
        res_time=0

        position=0
        counter=0.
        count_update_models=0.
        intermediate_models=[]
        intermediate_response=[]
        surrogate_lik=[]
        eikonal_lik=[]
        X_train=[]
        Y_train=[]
        res_time=0
        
        DESIGN_initial_PCE=scipy.io.loadmat('yFD_69tt_DESIGN_Loo_100_200.mat')
        an_array = np.array(list(DESIGN_initial_PCE.items()))
        X_init=an_array[3,1]
        Y_init=an_array[4,1]
        X_train_PCE_cum=X_init
        Y_train_PCE_cum=Y_init           
        temp_count=0.         

### Main sampling loop  
       

        while self.MCMCVar.Iter < self.MCMCPar.ndraw:
            start_time1c = time.time()
            
            if (self.MCMCVar.Iter < self.MCMCPar.tune_phi):  
               
                if (prov_AR < self.MCMCPar.AR_min):
                                
                    for i in range(MCMCPar.phi.shape[0]):
                    
                        self.MCMCPar.phi[i] = MCMCPar.phi[i]*0.8  
                        
            # Initialize totaccept
            totaccept = 0
         
            # Loop a number of K mcmc steps for each intermediate distribution

            for gen_number in range(0,self.MCMCPar.steps):
                
                time_complete_gen_start=time.time()
                # Update T
                self.MCMCVar.T = self.MCMCVar.T + 1
                
                for i in range(MCMCPar.phi.shape[0]):
                    phi_used[i]=MCMCPar.phi[i]      

                
                phi_ev=np.append(phi_ev,self.MCMCPar.phi.reshape(1,self.MCMCPar.seq),axis=0)
                
                phi_used_ev=np.append(phi_used_ev,phi_used.reshape(1,self.MCMCPar.seq),axis=0)                
                
                xold = np.array(self.MCMCVar.X[:self.MCMCPar.seq,:self.MCMCPar.n])
                log_p_xold = np.array(self.MCMCVar.X[:self.MCMCPar.seq,self.MCMCPar.n + 2-1])
                logprior_old= np.array(self.MCMCVar.X[:self.MCMCPar.seq,self.MCMCPar.n + 2])
                
               
                if (np.random.rand(1) <= self.MCMCPar.parallelUpdate):
                    Update = 'Parallel_Direction_Update'
                else:
                    Update = 'Snooker_Update'

                            
                out=np.zeros((self.MCMCPar.seq,self.MCMCPar.n))
                perturbation=np.zeros((self.MCMCPar.seq,self.MCMCPar.npca))
                xnew=np.zeros((self.MCMCPar.seq,self.MCMCPar.npca))

                njobs=self.MCMCPar.seq
               
                perturbation=np.random.multivariate_normal(np.zeros(self.MCMCPar.npca),(MCMCPar.matrix_std**2)*MCMCPar.phi[0],size=(self.MCMCPar.seq))           
                    
                xnew=xold+perturbation
                xnew_m=matlab.double(xnew.tolist())
       
                [YPCEm,prior_density]=eng.PCE_eval(xnew_m,myPCE,myInput,nargout=2) 
                YPCE=np.array(YPCEm)                

                of_xnew,log_p_xnew = CompLikelihood(xnew,YPCE,self.MCMCPar,self.Measurement,self.Extra,self.Measurement.total_cov)

                logprior_new=np.array(prior_density)
                
                if full_physics==True:     
                    xcurr_pr=eng.model_bprojection(xnew_m,self.MCMCPar.npca)            
                    xcurr_pr_p=np.asarray(xcurr_pr)
   
                    FF=RunFoward(xcurr_pr_p,self.MCMCPar,self.Measurement,self.ModelName,self.Extra)
                    of_FF,curr_log_lik_FF = CompLikelihood(xcurr_pr_p,FF,self.MCMCPar,self.Measurement,self.Extra,self.Measurement.data_cov)
                    log_p_xnew = curr_log_lik_FF
                    of_xnew=of_FF
                    YPCE=FF                

                # Calculate the Metropolis ratio
                accept = Metrop(self.MCMCPar,log_p_xnew,log_p_xold,logprior_new, logprior_old,Extra,beta)
                
                xnew=np.reshape(xnew,(self.MCMCPar.seq,self.MCMCPar.n))
                print('xnew-xold',np.sum(xnew-xold))
                # And update X and the model simulation
                idx_X= np.argwhere(accept==1);idx_X=idx_X[:,0]
  
                if not(idx_X.size==0):
                     
                    self.MCMCVar.X[idx_X,:] = np.concatenate((xnew[idx_X,:],of_xnew[idx_X,:],log_p_xnew[idx_X,:],logprior_new[idx_X,:]),axis=1)
                    self.MCMCVar.Xfx[idx_X,:] = YPCE[idx_X,:]

                
                # Check whether to add the current points to the chains or not?
                if self.MCMCVar.T == self.MCMCPar.thin:
                    # Store the current sample in Sequences
                    self.MCMCVar.iloc = self.MCMCVar.iloc + 1
                    aux=np.reshape(self.MCMCVar.X.T,(1,self.MCMCPar.n+3,self.MCMCPar.seq))
                    likelihood_ev=np.append(likelihood_ev,aux[0,self.MCMCPar.n+1,:])
                    self.Sequences[self.MCMCVar.iloc,:,:self.MCMCPar.seq] =aux[0,:,:]
                   
                    # Check whether to store the simulation results of the function evaluations
                    if self.MCMCPar.savemodout==True:
                        self.fx=np.append(self.fx,self.MCMCVar.Xfx,axis=0)
                        # Update m_func
                        self.MCMCVar.m_func = self.MCMCVar.m_func + self.MCMCPar.seq
                    else:
                        self.MCMCVar.m_func=None
                        # And set the T to 0
                    self.MCMCVar.T = 0

                counter=counter+1.
                
                ### Save intermediate results               
                if counter==1000:
                    intermediate_models=np.append(intermediate_models,self.MCMCVar.X)
                    intermediate_response=np.append(intermediate_response,YPCE)
                    np.save('intermediate_response.npy',intermediate_response)
                    np.save('intermediate_weights.npy',weig_seq)
                    np.save('intermediate_weights_increment.npy',weig_cont)
                    np.save('intermediate_models_run.npy',intermediate_models)                    
                    np.save('intermediate_beta.npy',beta_run)
                    np.save('intermediate_beta_surr.npy',beta_run_surr)
                    np.save('intermediate_AR.npy',OutDiag.AR)     
                    np.save('intermediate_CESS.npy',CESS_ev)           
                    np.save('intermediate_evid.npy',evid_ev)                        
                    np.save('intermediate_likelihood_ev',likelihood_ev)
                    np.save('intermediate_phi_ev',phi_used_ev)
                    np.save('intermediate_ESS',ESS_ev)                    
                    counter=0.
                    
                # Compute squared jumping distance for each CR value
                if (self.MCMCPar.Do_pCR==True and self.MCMCVar.Iter < 0.1 * self.MCMCPar.ndraw):
                   
                    # Calculate the standard deviation of each dimension of X
                    r = matlib.repmat(np.std(self.MCMCVar.X[:,:self.MCMCPar.n],axis=0),self.MCMCPar.seq,1)
                    # Compute the Euclidean distance between new X and old X
                    delta_normX = np.sum(np.power((xold[:,:self.MCMCPar.n] - self.MCMCVar.X[:,:self.MCMCPar.n])/r,2),axis=1)
                                        
                    # Use this information to update delta_tot which will be used to update the pCR values
                    self.MCMCVar.delta_tot = CalcDelta(self.MCMCPar.nCR,self.MCMCVar.delta_tot,delta_normX,self.MCMCVar.CR[:,gen_number])

                # Compute number of accepted moves
                totaccept = totaccept + np.sum(accept)

                # Update total number of MCMC iterations
                self.MCMCVar.Iter = self.MCMCVar.Iter + self.MCMCPar.seq
                
                time_complete_gen_end=time.time()
                print('time complete iteration',time_complete_gen_end-time_complete_gen_start)
                           
            print('salio del loop')

            # Store acceptance rate
            self.OutDiag.AR[self.MCMCVar.iteration-1,:] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), np.array([100 * totaccept/(self.MCMCPar.steps * self.MCMCPar.seq)]).reshape((1,1))),axis=1)
            
            prov_AR=100 * totaccept/(self.MCMCPar.steps * self.MCMCPar.seq)
            
            # Store probability of individual crossover values
            self.OutDiag.CR[self.MCMCVar.iteration-1,:self.MCMCPar.nCR+1] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), self.MCMCVar.pCR),axis=1)
            
            # Is pCR updating required?
            if (self.MCMCPar.Do_pCR==True): #and self.MCMCVar.Iter < 0.1 * self.MCMCPar.ndraw):

                # Update pCR values
                self.MCMCVar.pCR = AdaptpCR(self.MCMCPar.seq,self.MCMCVar.delta_tot,self.MCMCVar.lCR,self.MCMCVar.pCR)

            # Generate CR values from current pCR values
            self.MCMCVar.CR,lCRnew = GenCR(MCMCPar,self.MCMCVar.pCR); self.MCMCVar.lCR = self.MCMCVar.lCR + lCRnew

            # Calculate Gelman and Rubin Convergence Diagnostic
            start_idx = np.maximum(1,np.floor(0.5*self.MCMCVar.iloc)).astype('int64')-1; end_idx = self.MCMCVar.iloc
            
            # Update number of complete generation loops
            self.MCMCVar.iteration = self.MCMCVar.iteration + 1

            if self.MCMCPar.save_tmp_out==True:
                with open('out_tmp'+'.pkl','wb') as f:
                    pickle.dump({'Sequences':self.Sequences,
                    'OutDiag':self.OutDiag,'fx':self.fx,'MCMCPar':self.MCMCPar,
                    'MCMCVar':self.MCMCVar,'Measurement':self.Measurement,
                    'ModelName':self.ModelName,'Extra':self.Extra},f, protocol=pickle.HIGHEST_PROTOCOL)
 
            if beta>=1 and full_physics==True:
                curr_log_lik=np.array(self.MCMCVar.X[:self.MCMCPar.seq,self.MCMCPar.n + 2-1])                          
                last_models=self.MCMCVar.X           
                last_likelihoods=curr_log_lik.flatten()
                break
            
 
            if beta>=1 and full_physics==False: #now here I need to do the ISstep with full physics
                
                #variable para controlar
                
                full_physics=True
                
                curr_log_lik=np.array(self.MCMCVar.X[:self.MCMCPar.seq,self.MCMCPar.n + 2-1])   
            
                curr_x_models=np.array(self.MCMCVar.X[:self.MCMCPar.seq,:self.MCMCPar.n])   
            
                surrogate_lik=np.append(surrogate_lik,curr_log_lik)
            
                np.save('Final_surrogate_lik.npy',surrogate_lik)
                np.save('Final_models_beforeFF.npy',curr_x_models)
                np.save('Final_norm_weights_beforeFF.npy',norm_weight)                
                xcurr_m=matlab.double(curr_x_models.tolist())
       
                xcurr_pr=eng.model_bprojection(xcurr_m,self.MCMCPar.npca)
            
                xcurr_pr_p=np.asarray(xcurr_pr)
   
                FF=RunFoward(xcurr_pr_p,self.MCMCPar,self.Measurement,self.ModelName,self.Extra)

                of_FF,curr_log_lik_FF = CompLikelihood(xcurr_pr_p,FF,self.MCMCPar,self.Measurement,self.Extra,self.Measurement.data_cov)           
                eikonal_lik=np.append(eikonal_lik,curr_log_lik_FF)
                contribution_final = np.exp(curr_log_lik_FF.flatten()  - curr_log_lik.flatten())                 
#                np.save('eikonal_lik.npy',eikonal_lik)
                np.save('Final_contribution_lik.npy',contribution_final )   
                np.save('Final_eikonal_lik.npy',surrogate_lik)                
                
                next_beta,new_corr,CESS_found = search_update(curr_log_lik.flatten(),curr_log_lik_FF.flatten() ,CESSf,beta,norm_weight,self.MCMCPar.seq,full_physics) 
                
                np.save('new_corr.npy',new_corr)
                alpha_corrections=np.append(alpha_corrections,new_corr)
 
                np.save('TT_log_lik_surr_last.npy',curr_log_lik.flatten())
                np.save('TT_log_lik_surr_last_eikonal.npy',curr_log_lik_FF.flatten())
                np.save('TT_beta_surr_last.npy',beta)
                np.save('TT_norm_weights_last.npy',norm_weight)         
                
                self.MCMCVar.X[:self.MCMCPar.seq,self.MCMCPar.n + 2-1]= curr_log_lik_FF.flatten()     
                contribution = np.exp(next_beta * curr_log_lik_FF.flatten()   - beta * curr_log_lik.flatten())                  

            if beta<1 and full_physics==True:

                curr_log_lik=np.array(self.MCMCVar.X[:self.MCMCPar.seq,self.MCMCPar.n + 2-1])   
            
                curr_x_models=np.array(self.MCMCVar.X[:self.MCMCPar.seq,:self.MCMCPar.n])                   

                next_beta,incr,CESS_found,omega_new = binary_search(curr_log_lik,CESSf,beta,norm_weight,self.MCMCPar.seq,self.MCMCPar.gamma,omega)  

                contribution = np.exp((next_beta - beta) * curr_log_lik.flatten())                  
            
            if beta<1 and full_physics==False:
            
                curr_log_lik=np.array(self.MCMCVar.X[:self.MCMCPar.seq,self.MCMCPar.n + 2-1])   
            
                curr_x_models=np.array(self.MCMCVar.X[:self.MCMCPar.seq,:self.MCMCPar.n])   
            
                surrogate_lik=np.append(surrogate_lik,curr_log_lik)
            
                np.save('surrogate_lik.npy',surrogate_lik)
 
                xcurr_m=matlab.double(curr_x_models.tolist())
       
                xcurr_pr=eng.model_bprojection(xcurr_m,self.MCMCPar.npca)
            
                xcurr_pr_p=np.asarray(xcurr_pr)
   
                FF=RunFoward(xcurr_pr_p,self.MCMCPar,self.Measurement,self.ModelName,self.Extra)

                of_FF,curr_log_lik_FF = CompLikelihood(xcurr_pr_p,FF,self.MCMCPar,self.Measurement,self.Extra,self.Measurement.data_cov)
            
                eikonal_lik=np.append(eikonal_lik,curr_log_lik_FF)
            
                np.save('eikonal_lik.npy',eikonal_lik)
       
                next_beta,incr,CESS_found,omega_new = binary_search(curr_log_lik,CESSf,beta,norm_weight,self.MCMCPar.seq,self.MCMCPar.gamma,omega)  
 
                print('lo encontro')
                print('beta',next_beta)
 
                contribution = np.exp((next_beta - beta) * curr_log_lik.flatten())  
            
            #Save models and full physics simulations to train the surrogates:           
                temp_count=temp_count+1.
            
                np.save('temp_count.npy',temp_count)
            
                if temp_count==8. and update_surr<4.:
            #Save models and full physics simulations to train the surrogates:
                    np.save('temp_count_verif.npy',temp_count)           
                    xcurr_m=matlab.double(curr_x_models.tolist())
       
                    xcurr_pr=eng.model_bprojection(xcurr_m,self.MCMCPar.npca)
            
                    xcurr_pr_p=np.asarray(xcurr_pr)
   
                    FF=RunFoward(xcurr_pr_p,self.MCMCPar,self.Measurement,self.ModelName,self.Extra)

                    of_FF,curr_log_lik_FF = CompLikelihood(xcurr_pr_p,FF,self.MCMCPar,self.Measurement,self.Extra,self.Measurement.data_cov)
            
                    eikonal_lik=np.append(eikonal_lik,curr_log_lik_FF)
            
                    X_train=np.append(X_train,curr_x_models)
                    np.save('X_update.npy',X_train)
                    Y_train=np.append(Y_train,FF)
                    np.save('Y_eikonal_update.npy',Y_train)   
                    temp_count=0.
                
                           
                count_update_models=count_update_models+1.

                contribution = np.exp((next_beta - beta) * curr_log_lik.flatten())    

                   
 #               if count_update_models>=10: #Update the surrogate 
                if count_update_models==temp_surr_update and update_surr<4.:
                
                    temp_count=0.
                    update_surr=update_surr+1.
                             
                    count_update_models=int(count_update_models)   
               
                    X_train_size=int(X_train.shape[0]/MCMCPar.npca)
                    X_train=X_train.reshape((X_train_size,MCMCPar.npca))
                    Y_train=Y_train.reshape((X_train_size,MCMCPar.ypca))   
                    np.save('X_update_size'+str(update_surr)+'.npy',X_train.shape) 
                    np.save('Y_update_size'+str(update_surr)+'.npy',Y_train.shape)       
                    
                    samples_train=200
                    cumulative_samples=int(samples_train*(update_surr)+300) 
                    
                    extended=np.append(X_train,Y_train,axis=1)
                    extended_perm=np.random.permutation(extended) 
                    X_train_perm_cum=extended_perm[:,:MCMCPar.npca]
                    Y_train_perm_cum=extended_perm[:,MCMCPar.npca:MCMCPar.npca+MCMCPar.ypca]    
                    
                    X_update_PCE=X_train_perm_cum[:samples_train,:]
                    Y_update_PCE=Y_train_perm_cum[:samples_train,:]      
                    
                    X_train_PCE_cum=np.append(X_train_PCE_cum,X_update_PCE,axis=0)      
                    Y_train_PCE_cum=np.append(Y_train_PCE_cum,Y_update_PCE,axis=0)    

                    training_samples=int(X_train_PCE_cum.shape[0])

                    X_train_PCE_cum_dic = {"INPUT": X_train_PCE_cum}
                    scipy.io.savemat('X_train_PCE_cum_update.mat',X_train_PCE_cum_dic)        

                    Y_train_PCE_cum_dic = {"OUTPUT": Y_train_PCE_cum}            
                    scipy.io.savemat('Y_train_PCE_cum_update.mat',Y_train_PCE_cum_dic)
                                  
                    timeup=time.time()
                
                    myPCE=eng.PCE_update(MCMCPar.npca,MCMCPar.ypca,training_samples,nargout=1)

                    time2up=time.time()-timeup
                    print('time to re-train surrogate',time2up)
                
                    updated_total_cov=eng.get_cov(MCMCPar.npca,training_samples,myPCE,update_surr,nargout=1)              
                    mat3 = scipy.io.loadmat('UPDATED_COV'+str(int(update_surr))+'.mat')
                    an_array3 = np.array(list(mat3.items()))
                    total_cov = an_array3[3,1]
                    Measurement.total_cov = total_cov + Measurement.cov_pca
                
                    curr_x_models=matlab.double(curr_x_models.tolist())
                    [YPCEm,prior_density]=eng.PCE_eval(curr_x_models,myPCE,myInput,nargout=2) 
                    YPCE=np.array(YPCEm)                  
                    of_xnew_surr,curr_log_lik_new_surr = CompLikelihood(curr_x_models,YPCE,self.MCMCPar,self.Measurement,self.Extra,self.Measurement.total_cov)               

                    next_beta,new_corr,CESS_found = search_update(curr_log_lik.flatten(),curr_log_lik_new_surr.flatten() ,CESSf,beta,norm_weight,self.MCMCPar.seq,full_physics) 

                    np.save('new_corr.npy',new_corr)
                    alpha_corrections=np.append(alpha_corrections,new_corr)

                    X_train=[]
                    Y_train=[]
                    count_update_models=0.
 
                    contribution = np.exp(next_beta * curr_log_lik_new_surr.flatten()  - beta * curr_log_lik.flatten())         
                

                    np.save('TT_log_lik_surr_'+str(int(update_surr))+'.npy',curr_log_lik.flatten())
                    np.save('TT_log_lik_surr_up'+str(int(update_surr))+'.npy',curr_log_lik_new_surr.flatten())
                    np.save('TT_beta_surr_'+str(int(update_surr))+'.npy',beta)
                    np.save('TT_norm_weights'+str(int(update_surr))+'.npy',norm_weight)                
                    self.MCMCVar.X[:self.MCMCPar.seq,self.MCMCPar.n + 2-1]= curr_log_lik_new_surr.flatten()
            
            CESS_ev=np.append(CESS_ev,CESS_found)
            
            weig_cont=np.append(weig_cont,contribution)
            
            new_weight = np.multiply(norm_weight,contribution)
            
            new_weight_ev=np.append(new_weight_ev,new_weight)
            
            ESS=(np.sum(norm_weight*contribution))**2 / np.sum(norm_weight**2*contribution**2)
            
            ESS_ev=np.append(ESS_ev,ESS)
            
            norm_weight = new_weight / np.sum(new_weight)
            
            weig_seq=np.append(weig_seq,norm_weight)     
            
            weig_unn=np.multiply(weig_unn,contribution)
            
            weights_unnorm=np.append(weights_unnorm,weig_unn)
                   
            evid=np.sum(new_weight)
            
            evid_log=np.log(evid)
                        
            evid_cont=np.append(evid_cont, evid_log)
            
            evid_evolution=evid_evolution+evid_log
            
            evid_ev=np.append(evid_ev, evid_evolution)
            
### Perform resampling if needed
            
            if (ESS/self.MCMCPar.seq < MCMCPar.ESSf_div):
                
                res_time=res_time+1
                
                np.save('resampling_times.npy',res_time)
                
                print('Resample')
                
                Xres, ind, eve = resampling(norm_weight,self.MCMCPar.seq,self.MCMCVar.X,self.MCMCPar.n, anc_prev,eve_prev)
                
                eve_seq=np.append(eve_seq,eve)
                
                anc_prev=ind
                
                eve_prev=eve
                
                self.MCMCVar.X = Xres
                
                norm_weight= (1 / self.MCMCPar.seq) * np.ones(self.MCMCPar.seq)
              
            else:
                
                eve_seq=np.append(eve_seq,eve_prev)
              
            beta=next_beta
            omega=omega_new
            increment=np.append(increment,incr)
            beta_run=np.append(beta_run,beta)   
            
                
        start_time22 = time.time()
        # Remove zeros from pre-allocated variavbles if needed

        self.Sequences,self.OutDiag= Dreamzs_finalize(self.MCMCPar,self.Sequences,self.OutDiag,self.fx,self.MCMCVar.iteration,self.MCMCVar.iloc,self.MCMCVar.pCR)
       
        if self.MCMCPar.saveout==True:
            with open('dreamzs_out'+'.pkl','wb') as f:
                pickle.dump({'Sequences':self.Sequences,'OutDiag':self.OutDiag,'MCMCPar':self.MCMCPar,'Measurement':self.Measurement,'Extra':self.Extra},f
                , protocol=pickle.HIGHEST_PROTOCOL)
                
        end_time22 = time.time()
        print("This saving took %5.4f seconds." % (end_time22 - start_time22))


        self.Sequences=self.Sequences[1:,:,:]        

        eve_seq=eve_seq.reshape(int(eve_seq.shape[0]/self.MCMCPar.seq),self.MCMCPar.seq)
        
        return self.Sequences, self.OutDiag, self.MCMCPar, self.MCMCVar, beta_run, phi_used_ev, phi_ev, weig_seq, CESS_ev, increment, ESS_ev, evid_cont, evid_ev, weights_unnorm, new_weight_ev, weig_cont, eve_seq,last_models,last_likelihoods,intermediate_models,beta_run_surr,beta_K_steps,alpha_corrections
        
