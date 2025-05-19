import utils
import random
import numpy as np
import environment as env
from sarsa_dr_agent import Sarsa_DR

import os
import pandas as pd

actions = ["right", "left"]

if __name__ == "__main__":
    # Read arguments:
    args = utils.ArgsParser.read_input_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Instantiate objects I'll need
    environment = env.MDP(args.input)

    exp_dir = os.path.basename(__file__)[: -len(".py")]
    exp_dir = os.path.join("results", exp_dir)
    exp_dir = os.path.join(exp_dir, args.input.split("/")[-1].split(".")[0])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    df = pd.DataFrame(columns=["episode", "avg_undisc_return"])
    
    # Actual learning algorithm
    for ep in range(args.num_episodes):
        agent = Sarsa_DR(environment, args.step_size, args.step_size_dr, args.gamma, args.gamma_dr, 
                        args.lambda_dr, args.epsilon, args.beta, args.transform)
        time_step = 1
        while not environment.is_terminal():
            agent.step()
            time_step += 1
        environment.reset()
        
        df.loc[ep] = [ep, agent.get_avg_undisc_return()]
        print(ep, ",", agent.get_avg_undisc_return())
        
    df.loc[ep + 1] = ['train_avg', df['avg_undisc_return'].mean()]
    df.loc[ep + 2] = ['train_std', df['avg_undisc_return'].std()]
    df.loc[ep + 3] = ['train_max', df['avg_undisc_return'].max()]
    df.loc[ep + 4] = ['train_min', df['avg_undisc_return'].min()]
    df.loc[ep + 5] = ['train_median', df['avg_undisc_return'].median()]
    
    # Save the results
    df_name = f"{args.seed}_{args.step_size}_{args.step_size_dr}_{args.lambda_dr}_{args.epsilon}_{args.beta}.csv"
    df.to_csv(os.path.join(exp_dir, df_name), index=False)