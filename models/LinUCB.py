import numpy as np
from utils.oracle import *

def LinUCB(data_loader, v, oracle_func):
    '''
    data_loader: the data loader object
    v : exploration rate (tuning parameter)
    '''
    num_of_rounds, dim, k, N, mu = data_loader.return_info()

    V = np.eye(dim)
    b = np.zeros(dim)
    real_rewards = np.zeros(num_of_rounds)
    regrets = np.zeros(num_of_rounds)

    for t in range(num_of_rounds):
        sum_exp_optimal_reward, real_scores, contexts = data_loader.load_data_at_round_t(t)
        
        # update estimate and compute estimated scores
        mu_hat = np.linalg.inv(V).dot(b)
        est_reward = np.array([np.dot(context, mu_hat) for context in contexts])
        est_interval = np.array([np.sqrt(np.dot(np.dot(context, np.linalg.inv(V)), context)) for context in contexts])
        est_scores = est_reward + v * est_interval
        
        # get super set
        super_set = eval(oracle_func)
        
        # update parameter
        exp_reward = 0
        for act in super_set:
            V += np.outer(contexts[act], contexts[act])
            b += real_scores[act] * contexts[act]
            exp_reward += np.dot(contexts[act], mu)
        real_rewards[t] = data_loader.calculate_reward(super_set, t, evaluate=True)
        regrets[t] = sum_exp_optimal_reward - exp_reward
        
    return real_rewards, regrets