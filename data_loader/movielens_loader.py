import numpy as np
import random
import pandas as pd

class MovieLensDataLoader:
    def __init__(self, opts):
        self.opts = opts
        self.k = opts.super_set_size
        self.is_tuning = opts.is_tuning
        self.reward_function = opts.reward_function
        self.seed = opts.seed
        
        self.data = pd.read_csv('./data/ml-25m/preproc_data/test.csv')
        self.item_emb = np.load('./data/ml-25m/preproc_data/item_emb.npy')
        
        self.num_users = len(self.data['userId'].unique())
        self.num_items = self.item_emb.shape[0]
        
        self.tuning_T = opts.tuning_time_horizon
        self.num_of_rounds = self.tuning_T if self.is_tuning else self.num_users - self.tuning_T
        self.dim = self.item_emb.shape[1]
        
    def print_info(self):
        print('num_users:', self.num_users)
        print('num_items:', self.num_items)
        print('tuning_T:', self.tuning_T)
        print('T: ', self.num_users - self.tuning_T)
        
    def return_info(self):
        return self.num_of_rounds, self.dim, self.k, None, None
    
    def data_init_for_iter(self, iter):
        # in each iteration, we choose a random user
        random.seed(self.seed + iter)
        self.query_seq = random.sample(list(self.data['userId'].unique()), len(self.data['userId'].unique()))

        if self.is_tuning:
            self.query_seq = self.query_seq[:self.tuning_T]
        else:
            self.query_seq = self.query_seq[self.tuning_T:]

    
    def load_data_at_round_t(self, t):
        user = self.query_seq[t]
        user_data = self.data[self.data['userId'] == user]
        user_data = user_data.sort_values(by='rating', ascending=False)
        self.contexts = self.item_emb[user_data[:self.k]['movieId'].values]
        sum_optimal_reward = np.sum(user_data[:self.k]['rating'])
        self.real_score = np.array(user_data['rating'])
        
        if t % 1000 == 0:
            print(f"round {t} is done")
        return sum_optimal_reward, self.real_score, self.contexts
    
    def calculate_reward(self, arms, t, scores):
        return eval('self.' + self.reward_function)(arms, t, scores)

    def total_sum(self, super_set, t, scores):
        return np.sum([scores[arm] for arm in super_set])

    def diversity(self, super_set, t, scores):
        contexts = np.array([self.contexts[arm] for arm in super_set])
        reward = np.sum([scores[arm] for arm in super_set])
        reward += 1/2 * len(super_set) * np.log(2*np.pi*np.e) + 1/2 * np.log(np.linalg.det(np.dot(contexts, contexts.T)+np.eye(len(super_set))))
        return reward