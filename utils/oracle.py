import numpy as np

def set_oracle(oracle):
    if oracle == 'topk':
        return 'topk(est_scores, k)'
    elif oracle == 'greedy_oracle':
        return 'greedy_oracle(est_scores, data_loader, t)'
    elif oracle == 'greedy_oracle_for_diversity':
        return 'greedy_oracle_for_diversity(est_scores, data_loader, t)'
    

def topk(est_scores, k):
    return np.argsort(est_scores)[-k:]

# A greedy oracle algorithm that guarantees (1-e)-approximation when the reward is a nondecreasing submodular set function.
def greedy_oracle(est_scores, data_loader, t):
    _, dim, k, N = data_loader.return_info()
    As = set(range(N))
    S = set()
    for _ in range(k):
        rewards = np.zeros(N)
        for arm in set.difference(As, S):
            tmp_set = list(S)+[arm]
            rewards[arm] = data_loader.calculate_reward(tmp_set, t, est_scores)
        S.add(np.argmax(rewards))
    return np.array(list(S))

# A greedy oracle algorithm for rewards that reflects the diversity proposed in C2UCB.
def greedy_oracle_for_diversity(est_scores, data_loader, t, lamb = 0.2):
    '''
    lamb: the diversity parameter
    '''
    _, dim, k, N = data_loader.return_info()
    _, _, contexts = data_loader.load_data_at_round_t(t)
    
    sigma = 1
    C = sigma**(-2)
    As = set(range(N))
    S = set()
    XS = np.array(np.zeros(dim)).reshape(dim, -1)
    for _ in range(k):
        delta_gs = np.zeros(N)
        delta_Rs = np.zeros(N)
        for arm in set.difference(As, S):
            Sigma_iS = np.dot(contexts[arm], XS)
            delta_Rs[arm] = est_scores[arm]
            delta_gs[arm] = 1/2 * np.log(2 * np.pi * np.e * (sigma**2 + np.dot(Sigma_iS, C).dot(Sigma_iS.T)))
        S.add(np.argmax(delta_Rs + lamb * delta_gs))
        # XS = contexts.iloc[list(S),:].T
        XS = np.array([contexts[arm] for arm in S]).T
        C = np.linalg.inv(np.dot(XS.T, XS) + sigma**2 * np.eye(len(S)))
        
    return np.array(list(S))
