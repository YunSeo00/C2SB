
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

# plot
iterations = 5
error_var = 0.1
seed = 42
nu = 'set1'

time_horizon = 10000
tuning_time_horizon = 2000
reward_function = 'total_sum' # ['total_sum', 'diversity']
dataset = 'numerical'

models = "[('LinUCB','topk'),('LinTS','topk'),('C2SB','topk')]"
    
for dim in [2, 10]:
    for arms in [10, 20]:
        for super_set_size in [2, 4, 8]:
            if arms < super_set_size: continue
            for nu in ['set1', 'set2', 'set3', 'set4']:
                print(f"python utils/plot.py --dataset {dataset} --iteration {iterations} --tuning_time_horizon {tuning_time_horizon} --time_horizon {time_horizon} --error_var {error_var} --dim {dim} --arms {arms} --nu {nu} --reward_function {reward_function} --super_set_size {super_set_size} --seed {seed} --models \"{models}\" \n")
