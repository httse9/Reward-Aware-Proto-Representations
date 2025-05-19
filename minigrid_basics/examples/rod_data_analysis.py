import os
from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
from minigrid_basics.examples.plotter import Plotter, Colors
from itertools import product

rod_directory = join("minigrid_basics", "experiments", "rod")

# build the file name given the hyperparameters
def construct_filename(p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, seed):
    values = [p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options]
    values = [str(v) for v in values]
    filename = '-'.join(values) + f"-{seed}.pkl"
    return filename

# read data given env_name, representation, and hyperparameters
def read_data(env_name, representation, p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, seed=10):
    path = join(rod_directory, env_name, representation)

    all_rewards = []
    all_visit_percentage = []
    num_successful_seeds = 0

    for s in range(1, seed + 1):
        filename = construct_filename(p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, s)

        try:
            with open(join(path, filename), "rb") as f:
                data = pickle.load(f)

            all_rewards.append(data['rewards'])
            all_visit_percentage.append(data['visit_percentage'])
            num_successful_seeds += 1
        except:
            pass

    return num_successful_seeds < seed, np.array(all_rewards), np.array(all_visit_percentage)

# check how many seeds are present
def check_seed(env_name, representation, p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, seed=10):
    path = join(rod_directory, env_name, representation)

    successful_seeds = []

    for s in range(1, seed + 1):
        filename = construct_filename(p_option, dataset_size, learn_rep_iteration, representation_step_size, num_options, s)

        file_exists = os.path.isfile(join(path, filename))
        if file_exists:
            successful_seeds.append(s)
    
    seed_missing = len(successful_seeds) < seed  # whether some seeds are missing

    return seed_missing, successful_seeds

## hyperparameters
p_option = [0.01, 0.05, 0.1]
dataset_size = [100, 100000]
learn_rep_iter = [1, 10, 100]      
rep_lr = [0.01, 0.03, 0.1]
num_options = [1, 8, 1000]


def check_seed_env(env, rep):
    """
    Check for which hyperparameter settings some seeds are missing
    """
    for hyper in product(p_option, dataset_size, learn_rep_iter, rep_lr, num_options):
        seed_missing, ss = check_seed(env, rep, *hyper)

        if seed_missing:
            print("  ", ss, hyper, build_command(env, rep, *hyper))

def build_command(env, rep, p_option, dataset_size, learn_rep_iter, rep_lr, num_options):

    return f"sbatch --array=1-10 rod.sh {env} {rep} {p_option} {dataset_size} {learn_rep_iter} {rep_lr} {num_options}"

def compute_p_r_stat(env_name, representation):
    """
    For environment and representation (DR/SR) pair,
    read the data for all hyperparameters.
    Compute the avg. state-visitation and average reward.
    """

    ps = {}
    rs = {}
    for hyper in product(p_option, dataset_size, learn_rep_iter, rep_lr, num_options):
        seed_fail, r, p = read_data(env_name, representation, *hyper)

        hyper_strings = [str(v) for v in list(hyper)]
        hypername = '-'.join(hyper_strings)
        print(hypername)
        
        ps[hypername] = p
        rs[hypername] = r

    return ps, rs



if __name__ == "__main__":
    """
    """

    envs = ["dayan", "dayan_2", "fourrooms", "fourrooms_2", "gridroom", "gridroom_2", "gridmaze", "gridmaze_2", "gridroom_25", "gridmaze_29"]

    ### process data
    p_dict = {}
    r_dict = {}
    representation = ["SR", "DR"]

    for env_name in envs:
        p_dict[env_name] = {}
        r_dict[env_name] = {}

    for env_name in envs:
        # print(env_name)
        for rep in representation:
            # print(f"  {rep}")
            path = join(rod_directory, env_name, rep)

            try:    # try to read processed data if exists
                with open(join(path, "p.pkl"), "rb") as f:
                    p_dict[env_name][rep] = pickle.load(f)

                with open(join(path, "r.pkl"), "rb") as f:
                    r_dict[env_name][rep] = pickle.load(f)

            except: # process data and save

                ps, rs = compute_p_r_stat(env_name, rep)
            
                with open(join(path, "p.pkl"), "wb") as f:
                    pickle.dump(ps, f)

                with open(join(path, "r.pkl"), "wb") as f:
                    pickle.dump(rs, f)

                p_dict[env_name][rep] = ps
                r_dict[env_name][rep] = rs


