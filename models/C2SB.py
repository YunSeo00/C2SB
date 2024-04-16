import numpy as np
import scipy.stats

def C2SB(data_loader, v):
    '''
    data_loader: the data loader object
    v : exploration rate (tuning parameter)
    '''
    num_of_rounds, dim, k, N = data_loader.return_info()
    # optimal_reward = cum_reward + cum_regret
    real_rewards = np.zeros(num_of_rounds)
    regrets = np.zeros(num_of_rounds)
    y = np.zeros(dim)
    B = np.eye(dim)
    
    for t in range(num_of_rounds):
        optimal_reward, real_scores, contexts = data_loader.load_data_at_round_t(t)
        
        # update estimate and compute estimated scores
        mu_hat = np.linalg.inv(B).dot(y)
        
        V = (v**2) * np.linalg.inv(B) # variance of mu_tilde
        mu_tilde = np.random.multivariate_normal(mu_hat, V)
        est_scores = np.array([np.dot(context, mu_tilde) for context in contexts])
        
        # get super set
        super_set = np.argsort(est_scores)[-k:]
        
        # compute posterior dist
        mu_mc = scipy.stats.multivariate_normal.rvs(mu_tilde, V, 1000)
        est_mc = list((np.dot(contexts, mu_mc.T)).T)
        actions_mc = list(np.argsort(est_mc, axis=1)[:, -k:].flatten())
        pi_est = np.array([float(actions_mc.count(n)) / len(actions_mc) for n in range(N)])
        
        # update parameter
        b_mean = np.dot(pi_est.reshape(N, -1).T, contexts).flatten()
        for act in super_set:
            B += np.outer(contexts[act]-b_mean, contexts[act]-b_mean)
            y += 2 * (contexts[act] - b_mean) * real_scores[act]
            real_rewards[t] += real_scores[act]
        B += ((contexts - b_mean) * pi_est.reshape(N, -1)).T.dot(contexts - b_mean)
        regrets[t] = optimal_reward - real_rewards[t]
        
    return real_rewards, regrets