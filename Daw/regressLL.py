import numpy as np   
from Daw.q_learning import update_q
from Daw.policy_learning import update_tau
import Daw.make_choices as mkc

#estimate alpha, beta from equations update_q & make_choice
def regressLL_q(params, choice_hist, reward_hist):
    # Resave the initial parameter guesses
    alpha_param = params[0]
    beta_param = params[1]

    num_trials = len(choice_hist) + 1
    # Calculate the predicted values from the initial parameter guesses
    q_l = np.zeros((num_trials,1)) + 0.5
    q_r = np.zeros((num_trials,1)) + 0.5

        
    p = []
    for i in range(num_trials-1):
        #calculate p in current state that choice_hist is done!
        if(choice_hist[i] == 1):
            p.append(mkc.make_choice_log(beta_param, q_l[i], q_r[i]))
            q_l[i+1] = update_q(q_l[i], alpha_param, reward_hist[i])
            q_r[i+1] = q_r[i]
        else:
            p.append(mkc.make_choice_log(beta_param, q_r[i], q_l[i]))
            q_r[i+1] = update_q(q_r[i], alpha_param, reward_hist[i])
            q_l[i+1] = q_l[i]
            
    logLik = -np.sum(p)

    # Tell the function to return the NLL (this is what will be minimized)
    return(logLik)


##########################################################
#
#
##########################################################
def regressLL_complex_q(params, choice_hist, reward_hist):
    # Resave the initial parameter guesses
    alpha_param = params[0]
    beta_param = params[1]
    kappa_param = params[2]

    num_trials = len(choice_hist) + 1
    # Calculate the predicted values from the initial parameter guesses
    q_l = np.zeros((num_trials,1)) + 0.5
    q_r = np.zeros((num_trials,1)) + 0.5
    
    #make L / R
    l_arr = np.append(0, choice_hist)
    r_arr = np.append(0,np.ones(len(choice_hist)) - choice_hist)
        
    p = []
    for i in range(num_trials-1):
        #calculate p in current state that choice_hist is done!
        if(choice_hist[i] == 1):
            p.append(mkc.make_choice_log_complex(beta_param, q_l[i], q_r[i], l_arr[i], r_arr[i], kappa_param))
            q_l[i+1] = update_q(q_l[i], alpha_param, reward_hist[i])
            q_r[i+1] = q_r[i]
        else:
            p.append(mkc.make_choice_log_complex(beta_param, q_r[i], q_l[i], r_arr[i], l_arr[i], kappa_param))
            q_r[i+1] = update_q(q_r[i], alpha_param, reward_hist[i])
            q_l[i+1] = q_l[i]

    logLik = -np.sum(p)

    # Tell the function to return the NLL (this is what will be minimized)
    return(logLik)
    
    
    
#estimate alpha, beta from equations update_q & make_choice
def regressLL_p(params, choice_hist, reward_hist):
    # Resave the initial parameter guesses
    beta_param = params[0]

    num_trials = len(choice_hist) + 1
    tau_l = np.zeros((num_trials,1)) 
    tau_r = np.zeros((num_trials,1)) 
    
    reward_comparison = 0.5
    # equation 12
    update_tau = lambda tau_t0, reward : tau_t0 + (reward - reward_comparison)
    
        
    p = []
    for i in range(num_trials-1):
        #calculate p in current state that choice_hist is done!
        if(choice_hist[i] == 1):
            p.append(mkc.make_choice_log(beta_param, tau_l[i], tau_r[i]))
            tau_l[i+1] = update_tau(tau_l[i], reward_hist[i])
            tau_r[i+1] = tau_r[i]
        else:
            p.append(mkc.make_choice_log(beta_param, tau_r[i], tau_l[i]))
            tau_r[i+1] = update_tau(tau_r[i], reward_hist[i])
            tau_l[i+1] = tau_l[i]
            
    logLik = -np.sum(p)

    return(logLik)

