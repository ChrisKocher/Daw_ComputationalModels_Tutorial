import numpy as np


 #############
 #
 #
 #
 ##############  
def make_choice(beta, choice_target, choice_other):
    """
    Equation 3 in Daw´s tutorial
    """
    choice_prob_target = (np.exp(beta * choice_target))/ (
       np.exp(beta * choice_target) + np.exp(beta * choice_other))
    return(choice_prob_target)


 #############
 #
 #
 #
 ##############  
def make_choice_log(beta, choice_target, choice_other):
    """
    Equation 3 in Daw´s tutorial - log-form
    """
    choice_log = beta * choice_target - np.log(np.exp(beta * choice_target) + np.exp(beta * choice_other))
    
    return choice_log
    
    
 #############
 #
 #
 #
 ##############  
def make_choice_complex_log(beta, choice_target, choice_other, choice_helper_target, choice_helper_other, kappa):
    """
    Equation 13 in Daw´s tutorial
    """

    make_choice_log = (beta * choice_target + kappa * choice_helper_target ) \
        - np.log(np.exp(beta * choice_target + kappa * choice_helper_target)\
                 + np.exp(beta * choice_other + kappa * choice_helper_other))
        
    return make_choice_log