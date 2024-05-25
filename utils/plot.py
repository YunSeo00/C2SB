import numpy as np
import sys
import argparse
import os
import matplotlib.pyplot as plt

def read_options(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    # simulation setting
    parser.add_argument('--dataset', help='The dataset to use', type=str, default='numerical')
    parser.add_argument('--iterations', help='The number of iterations', type=int, default=5)
    parser.add_argument('--tuning_time_horizon', help='The length of tuning rounds', type=int, default=300)
    parser.add_argument('--time_horizon', help='The length of rounds', type=int, default=10000)
    parser.add_argument('--error_var', help='The variance of the error term', type=float, default=0.1)
    parser.add_argument('--dim', help='The dimension of the context', type=int, default=2)
    parser.add_argument('--arms', help='The number of arms', type=int, default=10)
    parser.add_argument('--nu', help='The parameter of the time-varying term', type=str, default='set3')
    parser.add_argument('--reward_function', help= 'The oracle to use', type=str, default='topK')
    parser.add_argument('--super_set_size', help='The size of the super set', type=int, default=4)
    parser.add_argument('--seed', help='The seed for the random number generator', type=int, default=42)
    parser.add_argument('--models', help='The models to graph', type=str, default='["LinUCB", "LinTS", "C2SB"]')
    parser.add_argument('--which_plot', help='The type of plot to generate', type=str, default='regret') # ['regret', 'reward']
    opts = parser.parse_args(args)
    return opts

opts = read_options(sys.argv[1:])

base_dir = os.getcwd()
if opts.dataset == 'numerical':
    save_path = f"dataset_{opts.dataset}_iteration_{opts.iterations}_tuningT_{opts.tuning_time_horizon}_T_{opts.time_horizon}_error_var_{str(opts.error_var).replace('.','')}_d_{opts.dim}_n_{opts.arms}_nu_{opts.nu}_reward_{opts.reward_function}_k_{opts.super_set_size}_seed_{opts.seed}"
else: # real data
    save_path = f"dataset_{opts.dataset}_iteration_{opts.iterations}_tuningT_{opts.tuning_time_horizon}_d_{opts.dim}_reward_{opts.reward_function}_k_{opts.super_set_size}_seed_{opts.seed}"
save_path = base_dir + '/results/' + save_path

if opts.dataset == 'numerical':
    save_name = f"dataset_{opts.dataset}_iteration_{opts.iterations}_T_{opts.time_horizon}_error_var_{str(opts.error_var).replace('.','')}_d_{opts.dim}_n_{opts.arms}_nu_{opts.nu}_reward_{opts.reward_function}_k_{opts.super_set_size}"
else:
    save_name = f"dataset_{opts.dataset}_iteration_{opts.iterations}_d_{opts.dim}_reward_{opts.reward_function}_k_{opts.super_set_size}"


fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.tight_layout(pad=4.0)

models = eval(opts.models)
colors = ['r','b','g','y','m','c','k','o','p']

for model, oracle in eval(opts.models):
    data = np.load(f'{save_path}/cum_{opts.which_plot}_{model}_{oracle}_{save_name}.npy')
    steps = np.arange(data.shape[1])
    
    model_mean = np.mean(data, axis=0)
    model_std = np.std(data, axis=0)
    color = colors.pop(0)
    
    axes[0].fill_between(steps, model_mean - 1.96 * model_std, model_mean + 1.96 * model_std, alpha=0.5, color=color)
    axes[0].plot(steps, model_mean, label=model, color=color)
    
    axes[1].fill_between(steps, np.percentile(data, 25, 0), np.percentile(data, 75, 0), alpha=0.5, color=color)
    axes[1].plot(steps, np.median(data, axis=0), label=model, color=color)
    
axes[0].set_title('Cumulative Regret with mean and 95% confidence interval')
axes[0].set_xlabel('Decision Point')
axes[0].set_ylabel('Cumulative Regret')
axes[0].legend(loc='lower right')

axes[1].set_xlabel('Decision Point')
axes[1].set_ylabel('Cumulative Regret')
axes[1].set_title('Cumulative Regret with median and Q1, Q3')
axes[1].legend(loc='lower right')

plt.savefig(f'{save_path}/{save_name}.png')