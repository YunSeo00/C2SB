import numpy as np

class NumericalDataLoader:
    def __init__(self, opts):
        self.opts = opts
        self.dim = opts.dim
        self.N = opts.arms
        self.k = opts.super_set_size
        self.R = opts.error_var
        self.nu = opts.nu
        self.is_tuning = opts.is_tuning
        self.iter_num = opts.iterations
        self.T = opts.time_horizon
        self.tuningT = opts.tuning_time_horizon
        self.num_of_rounds = self.tuningT if self.is_tuning else self.T
        self.seed = opts.seed
        
    def return_info(self):
        return self.num_of_rounds, self.dim, self.k, self.N
    
    def data_init_for_iter(self, iter):
        self.data = list()
        self.sum_real_optimal_rewards = np.zeros(self.num_of_rounds)
        self.real_scores = list()
        self.vs = np.zeros(self.num_of_rounds)
        
        # set base seed
        if self.is_tuning:
            seed = self.T * self.iter_num + self.tuningT * iter + self.seed
        else:
            seed = self.T * iter + self.seed
            
        self.mu = self.make_mu(seed) # create true mu for each iteration
        
        # create data for each iteration
        for t in range(self.num_of_rounds):
            np.random.seed(seed + t)
            
            self.data.append(self.make_selectable_arms_data())
            contexts = self.data[t][0]
            errors = self.data[t][1]
            
            expected_scores = [np.dot(context, self.mu) for context in contexts]
            actions = np.argsort(expected_scores)[-self.k:]
            
            # compute nu(t), the nonparametric intercepy terms (not used method because this calculation uses expected_scores)
            if self.nu == 'set1': self.vs[t] = 0
            elif self.nu == 'set2': self.vs[t] = -expected_scores[actions[0]]
            elif self.nu == 'set3': self.vs[t] = np.log(t+1)
            elif self.nu == 'set4': self.vs[t] = np.cos(t * np.pi / 5000) * np.log(t + 1)
            elif self.nu == 'set5': self.vs[t] = np.cos(t * np.pi / 50) * np.log(t + 1)
            elif self.nu == 'set6': self.vs[t] = np.log2(t+1) * np.sin(0.0005 * t) ** 2 + t ** (1/4)
            elif self.nu == "set7": self.vs[t] = - np.cos(0.0005 * t) * np.sqrt(np.abs(expected_scores[actions[0]]))
            elif self.nu == "set8": self.vs[t] = np.sqrt(np.log(t+1))
            elif self.nu == "set9": self.vs[t] = (np.sin(0.0005 * t) * t ** (1/2))/(np.log(t+2))
            elif self.nu == "set10": self.vs[t] = (np.sin(0.0005 * t) ** 2 * t ** (1/2))/(np.log(t+2))
            elif self.nu == "set11": self.vs[t] = (np.sin(t / 20) * t ** (1/2))/(np.log(t+2))
            
            self.real_scores.append(np.array([np.dot(context, self.mu) + error + self.vs[t] for context, error in zip(contexts, errors)])) # contexts \cdot mu + nu_t + error_t
            actions = np.argsort(self.real_scores[t])[-self.k:]
            self.sum_real_optimal_rewards[t] = np.sum(self.real_scores[t][actions])  # \sum_{i \in S} (contexts \cdot mu + nu_t + error_t)
            
            # if t % 500 == 0:
            #     print(self.real_scores[t][actions])
            #     print(self.sum_real_optimal_rewards[t])

    def load_data_at_round_t(self, t):
        return self.sum_real_optimal_rewards[t], self.real_scores[t], self.data[t][0] # optimal_reward, real_scores, contexts
    
    def make_mu(self, seed) -> np.array: # 각 iteration 마다 1번씩 호출
        np.random.seed(seed)
        mu = np.random.uniform(0, 1, self.dim)
        euclidean = np.sqrt(np.sum(mu**2))
        mu = mu / euclidean
        return mu
    
    def make_x_without_intercept(self) -> np.array:
        x = np.random.uniform(0, 1, self.dim)
        euclidean = np.sqrt(np.sum(x**2))
        x = x / euclidean
        return x
    
    def make_x_with_intercept(self) -> np.array:
        x = np.random.uniform(0, 1, self.dim)
        euclidean = np.sqrt(np.sum(x**2))
        x = x / euclidean
        x = np.append(x, 1 / self.dim)
        return x
    
    def make_selectable_arms_data(self, intercept=False) -> list:
        contexts = list()
        errors = list()
        for i in range(self.N):
            if intercept:
                contexts.append(self.make_x_with_intercept())
            else:
                contexts.append(self.make_x_without_intercept())
            errors.append(np.random.normal(0, self.R))
        return [contexts, errors]

    
        
# def load_vs(nu_type):
#     pass

# def make_mu():
#     pass

# def make_x():
#     pass

# def load_data():
#     pass