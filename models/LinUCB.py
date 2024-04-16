import numpy as np

def LinUCB(data_loader, v):
    '''
    data_loader: the data loader object
    v : exploration rate (tuning parameter)
    '''
    num_of_rounds, dim, k, N = data_loader.return_info()

    V = np.eye(dim)
    b = np.zeros(dim)
    real_rewards = np.zeros(num_of_rounds)
    regrets = np.zeros(num_of_rounds)

    for t in range(num_of_rounds):
        optimal_reward, real_scores, contexts = data_loader.load_data_at_round_t(t)
        
        # update estimate and compute estimated scores
        mu_hat = np.linalg.inv(V).dot(b)
        est_reward = np.array([np.dot(context, mu_hat) for context in contexts])
        est_interval = np.array([np.sqrt(np.dot(np.dot(context, np.linalg.inv(V)), context)) for context in contexts])
        est_upper = est_reward + v * est_interval
        
        # get super set
        actions = np.argsort(est_upper)[-k:]
        
        # update parameter
        for act in actions:
            V += np.outer(contexts[act], contexts[act])
            b += real_scores[act] * contexts[act]
            real_rewards[t] += real_scores[act]
        regrets[t] = optimal_reward - real_rewards[t]
        
    return real_rewards, regrets