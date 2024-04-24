
# numerical data simulation (for set1~set11)

iterations = 5
error_var = 0.1
seed = 42

time_horizon = 50000
tuning_time_horizon = 5000
dim = 2
arms = 10
super_set_size = 4
reward_function = 'diversity' # ['total_sum', 'diversity']
dataset = 'numerical'

models = "[('LinUCB','topk'),('LinTS','topk'),('C2SB','topk')]"
is_tuning = True

for nu in ['set1', 'set2', 'set3', 'set4', 'set5', 'set6', 'set7', 'set8', 'set9', 'set10', 'set11']:
    save_path = f"dataset_{dataset}_iteration_{iterations}_tuningT_{tuning_time_horizon}_T_{time_horizon}_error_var_{str(error_var).replace('.','')}_d_{dim}_n_{arms}_nu_{nu}_reward_{reward_function}_k_{super_set_size}_seed_{seed}"
    if is_tuning:
        log_save_name = f'log/log_tuning_{save_path}.log'
    else:
        log_save_name = f'log/log_{save_path}.log'
    print(f"nohup python main.py --dataset {dataset} --iteration {iterations} --tuning_time_horizon {tuning_time_horizon} --time_horizon {time_horizon} --error_var {error_var} --dim {dim} --arms {arms} --nu {nu} --reward_function {reward_function} --super_set_size {super_set_size} --seed {seed} --models \"{models}\" --is_tuning {is_tuning} 1>{log_save_name} 2>&1 &")