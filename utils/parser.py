import argparse
import sys

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_options(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    # simulation setting
    parser.add_argument('--iterations', help='The number of iterations', type=int, default=5)
    parser.add_argument('--time_horizon', help='The length of rounds', type=int, default=50)
    parser.add_argument('--error_var', help='The variance of the error term', type=float, default=0.1)
    parser.add_argument('--dim', help='The dimension of the context', type=int, default=2)
    parser.add_argument('--arms', help='The number of arms', type=int, default=20)
    parser.add_argument('--super_set_size', help='The size of the super set', type=int, default=4)
    parser.add_argument('--nu', help='The parameter of the time-varying term', type=str, default='set3')
    parser.add_argument('--is_tuning', help='Whether to tune the parameter', type=str2bool, default=True)
    parser.add_argument('--tuning_time_horizon', help='The length of tuning rounds', type=int, default=3000)
    parser.add_argument('--reward_function', help= 'The reward function to use', type=str, default='diversity') # ['total_sum', 'diversity']
    parser.add_argument('--dataset', help='The dataset to use', type=str, default='numerical')
    parser.add_argument('--seed', help='The seed for the random number generator', type=int, default=42)
    # model
    parser.add_argument('--models', help='The model and oracle set (ex. [(model_name, oracle_type), ...])', type=str, 
                        default='[("LinUCB", "topk"), ("LinUCB", "greedy_oracle"), ("LinUCB", "greedy_oracle_for_diversity"), \
                                  ("LinTS", "topk"), ("LinTS", "greedy_oracle"), ("LinTS", "greedy_oracle_for_diversity"), \
                                  ("C2SB", "topk"), ("C2SB", "greedy_oracle"), ("C2SB", "greedy_oracle_for_diversity")]') 
    # [("LinUCB", "topk"), ("LinTS", "topk"), ("C2SB", "topk"), ("C2SB", "greedy_oracle"), ("C2SB", "greedy_oracle_for_diversity")]
    
    opts = parser.parse_args(args)
    return opts