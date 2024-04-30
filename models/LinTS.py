import numpy as np
from utils.oracle import *

def LinTS(data_loader, v, oracle_func):
    '''
    data_loader: the data loader object
    v : exploration rate (tuning parameter)
    '''
    num_of_rounds, dim, k, N, mu = data_loader.return_info()

    B = np.eye(dim)
    y = np.zeros(dim)
    real_rewards = np.zeros(num_of_rounds)
    regrets = np.zeros(num_of_rounds)
    
    for t in range(num_of_rounds):
        sum_exp_optimal_reward, real_scores, contexts = data_loader.load_data_at_round_t(t)
        
        # update estimate and compute estimated scores
        mu_hat = np.linalg.inv(B).dot(y)
        V = (v**2) * np.linalg.inv(B)
        mu_tilde = np.random.multivariate_normal(mu_hat, V)
        est_scores = np.array([np.dot(context, mu_tilde) for context in contexts])
        
        # get super set
        super_set = eval(oracle_func)
        
        # update parameter
        exp_reward = 0
        for act in super_set:
            B += np.outer(contexts[act], contexts[act])
            y += real_scores[act] * contexts[act]
            exp_reward += np.dot(contexts[act], mu)
        real_rewards[t] = data_loader.calculate_reward(super_set, t, evaluate=True)
        regrets[t] = sum_exp_optimal_reward - exp_reward
        
    return real_rewards, regrets