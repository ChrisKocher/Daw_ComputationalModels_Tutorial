import numpy as np

def sample_subjects(alpha_mean, alpha_std, alpha_low, alpha_up, beta_mean, beta_std, beta_low, beta_up, num_subjects):
    
    alpha_support_lower = alpha_low
    alpha_support_upper = alpha_up
    alpha_subjects_list = []
    
    beta_support_lower = beta_low
    beta_support_upper = beta_up
    beta_subjects_list = []
    
    while(len(alpha_subjects_list) < num_subjects):
        draw = np.random.normal(loc=alpha_mean, scale = alpha_std)
        if(draw <= alpha_support_upper and draw >= alpha_support_lower):
            alpha_subjects_list.append(draw)
        
    while(len(beta_subjects_list) < num_subjects):
        draw = np.random.normal(loc=beta_mean, scale = beta_std)
        if(draw <= beta_support_upper and draw >= beta_support_lower):
            beta_subjects_list.append(draw)
            
    alpha_subjects = np.asarray(alpha_subjects_list)
    beta_subjects = np.asarray(beta_subjects_list)
    
   return alpha_subjects, beta_subjects