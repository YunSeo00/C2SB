import numpy as np
import scipy.stats
from utils.oracle import *

def C2SB(data_loader, v, oracle_func):
    '''
    data_loader: the data loader object
    v : exploration rate (tuning parameter)
    '''
    num_of_rounds, dim, k, N, mu = data_loader.return_info()
    # optimal_reward = cum_reward + cum_regret
    real_rewards = np.zeros(num_of_rounds)
    regrets = np.zeros(num_of_rounds)
    y = np.zeros(dim)
    B = np.eye(dim)
    
    for t in range(num_of_rounds):
        sum_optimal_reward, real_scores, contexts = data_loader.load_data_at_round_t(t)
        if mu is None:
            N = len(real_scores)
        
        # update estimate and compute estimated scores
        mu_hat = np.linalg.inv(B).dot(y)
        
        V = (v**2) * np.linalg.inv(B) # variance of mu_tilde
        mu_tilde = np.random.multivariate_normal(mu_hat, V)
        est_scores = np.array([np.dot(context, mu_tilde) for context in contexts])
        
        # get super set
        # super_set = np.argsort(est_scores)[-k:]
        super_set = eval(oracle_func)
        
        # compute posterior dist
        mu_mc = scipy.stats.multivariate_normal.rvs(mu_tilde, V, 1000)
        est_mc = list((np.dot(contexts, mu_mc.T)).T)
        actions_mc = list(np.argsort(est_mc, axis=1)[:, -k:].flatten())
        pi_est = np.array([float(actions_mc.count(n)) / len(actions_mc) for n in range(N)])
        
        # update parameter
        b_mean = np.dot(pi_est.reshape(N, -1).T, contexts).flatten()
        exp_reward = 0
        for act in super_set:
            B += np.outer(contexts[act]-b_mean, contexts[act]-b_mean)
            y += 2 * (contexts[act] - b_mean) * real_scores[act]
            if mu is not None: # data is numerical data
                exp_reward += np.dot(contexts[act], mu)
        real_rewards[t] = data_loader.calculate_reward(super_set, t, real_scores)
        if mu is not None:
            regrets[t] = sum_optimal_reward - exp_reward # sum_optimal_reward is expected reward
        else:
            regrets[t] = sum_optimal_reward - real_rewards[t] # sum_optimal_reward is real reward
    
    return real_rewards, regrets