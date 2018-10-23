import numpy as np
from Daw.q_learning import update_q
import Daw.make_choices as mkc

def p_grid(len_alpha, lower_alpha, upper_alpha, len_beta, lower_beta, upper_beta, choice_hist, reward_hist):
    """
    PUT DOCU HERE
    """
    alpha_params = np.linspace(lower_alpha, upper_alpha,len_alpha)
    beta_params = np.linspace(lower_beta, upper_beta, len_beta)
    p_grid = np.zeros((len_alpha, len_beta))
    
    num_trials = len(choice_hist) + 1
    
    for i in range(len_alpha):
        for j in range(len_beta):

            alpha_param = alpha_params[i]
            beta_param = beta_params[j]

            # Calculate the predicted values from the initial parameter guesses
            q_l = np.zeros((num_trials,1)) + 0.5
            q_r = np.zeros((num_trials,1)) + 0.5

            p_temp = []
            for k in range(num_trials-1):
                if(choice_hist[k] == 1):
                    p_temp.append(mkc.make_choice_log(beta_param, q_l[k], q_r[k]))
                    q_l[k+1] = update_q(q_l[k], alpha_param, reward_hist[k])
                    q_r[k+1] = q_r[k]
                else:
                    p_temp.append(mkc.make_choice_log(beta_param, q_r[k], q_l[k]))
                    q_r[k+1] = update_q(q_r[k], alpha_param, reward_hist[k])
                    q_l[k+1] = q_l[k]

            p_grid[i][j] = -np.sum(p_temp)
    return p_grid