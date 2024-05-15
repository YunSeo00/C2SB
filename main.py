import argparse
import pandas as pd
import numpy as np
import sys
import os

from utils.parser import read_options

from models.LinUCB import LinUCB
from models.LinTS import LinTS
from models.C2SB import C2SB
from utils.oracle import set_oracle



opts = read_options(sys.argv[1:])

base_dir = os.getcwd()
save_path = f"dataset_{opts.dataset}_iteration_{opts.iterations}_tuningT_{opts.tuning_time_horizon}_T_{opts.time_horizon}_error_var_{str(opts.error_var).replace('.','')}_d_{opts.dim}_n_{opts.arms}_nu_{opts.nu}_reward_{opts.reward_function}_k_{opts.super_set_size}_seed_{opts.seed}"

if save_path is not None:
    save_path = base_dir + '/results/' + save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path + '/args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(opts).items(), key=lambda x: x[0])]))
        f.write('\n')


# load data_loader
if opts.dataset == 'numerical':
    from data_loader.numerical_loader import *
    data_loader = NumericalDataLoader(opts)

print("This simulation is running with the following options:")
print(*opts.__dict__.items(), sep="\n")

exploitation_rate_tuning_list = [1, 0.5, 0.25, 0.125, 0.0625, 0.0125, 0.00625]
tuning_num = len(exploitation_rate_tuning_list)

if opts.is_tuning == True:
    save_name = f"dataset_{opts.dataset}_iteration_{opts.iterations}_tuningT_{opts.tuning_time_horizon}_error_var_{str(opts.error_var).replace('.','')}_d_{opts.dim}_n_{opts.arms}_nu_{opts.nu}_reward_{opts.reward_function}_k_{opts.super_set_size}"
    print(save_name)    
    # opt_cum_expected_reward, opt_cum_real_reward = optimal()
    for model, oracle in eval(opts.models):
        oracle_func = set_oracle(oracle)
        
        print(f"Running {model}_{oracle} with tuning")
        tuning_table = pd.DataFrame(columns=['exploitation_rate'])
        tuning_table['exploitation_rate'] = exploitation_rate_tuning_list
        tuning_table['cumulative_real_reward'] = np.nan
        for exploitation_rate in exploitation_rate_tuning_list:
            real_reward_list = []
            for iter in range(opts.iterations):
                data_loader.data_init_for_iter(iter)
                
                real_rewards, _ = eval(model)(data_loader, exploitation_rate, oracle_func)
                real_reward_list.append(np.cumsum(real_rewards)[-1])
            tuning_table.loc[tuning_table['exploitation_rate'] == exploitation_rate, 'cumulative_real_reward'] = np.mean(real_reward_list)
        tuning_table.to_csv(f"{save_path}/tuning_real_reward_{model}_{oracle}_{save_name}.csv", index=False)

# when tuning is not needed
else:
    tuning_save_name = f"dataset_{opts.dataset}_iteration_{opts.iterations}_tuningT_{opts.tuning_time_horizon}_error_var_{str(opts.error_var).replace('.','')}_d_{opts.dim}_n_{opts.arms}_nu_{opts.nu}_reward_{opts.reward_function}_k_{opts.super_set_size}"
    save_name = f"dataset_{opts.dataset}_iteration_{opts.iterations}_T_{opts.time_horizon}_error_var_{str(opts.error_var).replace('.','')}_d_{opts.dim}_n_{opts.arms}_nu_{opts.nu}_reward_{opts.reward_function}_k_{opts.super_set_size}"
    
    for model, oracle in eval(opts.models):
        oracle_func = set_oracle(oracle)

        print(f"Running {model}_{oracle} main simulation")
        # select optimal exploitation rate
        tuning_table = pd.read_csv(f"{save_path}/tuning_real_reward_{model}_{oracle}_{tuning_save_name}.csv")
        exploitation_rate = tuning_table.loc[tuning_table['cumulative_real_reward'].idxmax()]['exploitation_rate']
        print(f"Optimal exploitation rate for {model}_{oracle} is {exploitation_rate}")
        
        # run the main simulation
        cum_regret_list = []
        cum_reward_list = []
        for iter in range(opts.iterations):
            data_loader.data_init_for_iter(iter)
            
            real_rewards, regrets = eval(model)(data_loader, exploitation_rate, oracle_func)
            cum_reward_list.append(np.cumsum(real_rewards))
            cum_regret_list.append(np.cumsum(regrets))
        # pd.DataFrame(cum_reward_list).to_csv(f"{save_path}/cum_reward_{model}_{save_name}.csv", index=False)
        # pd.DataFrame(cum_regret_list).to_csv(f"{save_path}/cum_regret_{model}_{save_name}.csv", index=False)        
        with open(f"{save_path}/cum_reward_{model}_{oracle}_{save_name}.npy", "wb") as f:
            np.save(f, cum_reward_list)
        with open(f"{save_path}/cum_regret_{model}_{oracle}_{save_name}.npy", "wb") as f:
            np.save(f, cum_regret_list)