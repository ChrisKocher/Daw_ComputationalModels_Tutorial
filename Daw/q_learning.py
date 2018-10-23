import numpy as np
#from make_choice import make_choice
import Daw.make_choices as mkc

def q_learning(alpha, beta, num_trials):
    
    q_l = np.zeros((num_trials,1)) + 0.5
    q_r = np.zeros((num_trials,1)) + 0.5

    choice_hist = []
    reward_hist = []

    for i in range(num_trials-1):
        current_choice = mkc.make_choice(beta, q_l[i], q_r[i])

        #draw random variable for making choice selection
        temp_rand = np.random.rand()
        
        #left
        if(temp_rand <= current_choice):
            reward = np.random.choice([0,1], p = [0.4, 0.6])
            reward_hist.append(reward)
            q_l[i+1] = update_q(q_l[i], alpha, reward)
            q_r[i+1] = q_r[i]
            #choice 1 = left
            choice_hist.append(1)
            
        #right
        else:
            reward = np.random.choice([0,1], p = [0.6, 0.4])
            reward_hist.append(reward)
            q_r[i+1] = update_q(q_r[i], alpha, reward)
            q_l[i+1] = q_l[i]
            choice_hist.append(0)
    
    return choice_hist, reward_hist, q_l, q_r


def update_q(q_choice_t0, alpha, reward):
    """
    equation 2 in Daw´s tutorial
    """
    q_updated = q_choice_t0 + alpha * (reward - q_choice_t0)
    return q_updated
