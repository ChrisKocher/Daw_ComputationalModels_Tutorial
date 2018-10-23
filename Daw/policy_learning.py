import numpy as np
import Daw.make_choices as mkc


def policy_learning(beta, num_trials):
    import numpy as np
    tau_l = np.zeros((num_trials,1)) 
    tau_r = np.zeros((num_trials,1)) 
    
    #cf page xx 
    reward_comparison = 0.5
       
    choice_hist = []
    reward_hist = []

    for i in range(num_trials-1):
        current_choice = mkc.make_choice(beta, tau_l[i], tau_r[i])
        
        temp_rand = np.random.rand()
        #left
        if(temp_rand <= current_choice):
            # previous: [0,1], p=[0.4,0.6]
            reward = np.random.choice([0,1], p = [0.5, 0.5])
            reward_hist.append(reward)
            tau_l[i+1] = update_tau(tau_l[i], reward, reward_comparison)
            tau_r[i+1] = tau_r[i]
            #choice 1 = left
            choice_hist.append(1)
        #right
        else:
            reward = np.random.choice([0,1], p = [0.5, 0.5])
            reward_hist.append(reward)
            tau_r[i+1] = update_tau(tau_r[i], reward, reward_comparison)
            tau_l[i+1] = tau_l[i]
            choice_hist.append(0)
        
    return choice_hist, reward_hist, tau_l, tau_r
    
    
def update_tau(tau_t0, reward, reward_comparison):
    """
    Equation 12 in Daw´s tutorial
    """
    
    updated_tau = tau_t0 + (reward - reward_comparison)
    
    return updated_tau