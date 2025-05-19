from minigrid_basics.examples.ROD_cycle import RODCycle
import os
import numpy as np
from flint import arb_mat, ctx
from itertools import islice

ctx.dps = 100   # important

# testing imports
import gym
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import subprocess
import glob
import pickle

def power_iteration(A, num_iters=1000, tol=1e-9):
    b = np.random.rand(A.shape[1])
    b = b / np.linalg.norm(b)

    for _ in range(num_iters):
        b_new = A @ b
        b_new = b_new / np.linalg.norm(b_new)
        if np.linalg.norm(b_new - b) < tol:
            break
        b = b_new

    return b

class RODCycle_DR(RODCycle):

    def __init__(self, env, n_steps=100, p_option=0.05, dataset_size=None, learn_rep_iteration=10, representation_step_size=0.1,
                 gamma=0.99, num_options=None, eigenoption_step_size=0.1, lambd=1.3, plot=True):
        super().__init__(env, n_steps=n_steps, p_option=p_option, dataset_size=dataset_size, learn_rep_iteration=learn_rep_iteration,
            representation_step_size=representation_step_size, gamma=gamma, num_options=num_options, eigenoption_step_size=eigenoption_step_size,
            plot=plot)
        
        self.lambd = lambd # lambda for the DR
        self.plot_path = "minigrid_basics/DR_ROD"
        
        self.reset()
        

    def reset(self):
        """
        Initialize DR TD learning
        """
        super().reset()
        self.representation = np.eye(self.env.num_states)


    def learn_representation(self):
        """
        DR TD learning
        """
        if self.dataset_size is not None:
            dataset = self.dataset[-self.dataset_size:]
        else:
            dataset = self.dataset

        # do one backward pass through dataset for theoretical guarantee
        for (s, a, r, ns) in reversed(dataset):
            indicator = np.zeros((self.env.num_states))
            indicator[s] = 1
            self.representation[s] += self.representation_step_size * (np.exp(r / self.lambd) * (indicator + self.representation[ns]) - self.representation[s])

        # remaining iterations, do forward pass
        for _ in range(self.learn_rep_iteration - 1):
            for (s, a, r, ns) in dataset:

                indicator = np.zeros((self.env.num_states))
                indicator[s] = 1

                self.representation[s] += self.representation_step_size * (np.exp(r / self.lambd) * (indicator + self.representation[ns]) - self.representation[s])


    def compute_eigenvector(self):
        """
        
        """
        DR = (self.representation + self.representation.T) / 2

        # get idx of visited states
        visited_idx = (DR.sum(1) != 1)
        DR_visited = DR[visited_idx][:, visited_idx]

        # do eigendecomposition only on visited states for stability
        DR_visited = arb_mat(DR_visited.tolist())
        lamb_visited, e_visited = DR_visited.eig(right=True, algorithm="approx", )
        lamb_visited = np.array(lamb_visited).astype(np.clongdouble).real.flatten()
        e_visited = np.array(e_visited.tolist()).astype(np.clongdouble).real.astype(np.float32)

        # sort eigenvalue and eigenvectors
        idx = np.argsort(lamb_visited)
        lamb_visited = lamb_visited[idx]
        e_visited = e_visited.T[idx]

        # get top eigenvector, assert same sign for all entries
        e0_visited = e_visited[-1]
        # if numerical issues persits, use power iteration
        if not ((e0_visited < 0).all() or (e0_visited > 0).all()):
            DR_visited = DR[visited_idx][:, visited_idx]
            e0_visited = power_iteration(DR_visited)

        assert (e0_visited < 0).all() or (e0_visited > 0).all()

        # project back to full state space
        e0 = np.zeros_like(visited_idx).astype(float)
        e0[visited_idx] = e0_visited

        # take log
        if e0.sum() < 0:
            e0 *= -1
        log_e0 = np.where(e0 > 0, np.log(e0), e0)       # apply log only on positive entries

        # normalize
        if (log_e0 != 0).any():
            log_e0 /= np.sqrt(log_e0 @ log_e0)
        else:
            log_e0 += 1 / np.sqrt(self.env.num_states)
        assert np.isclose(log_e0 @ log_e0, 1.0)

        return log_e0



if __name__ == "__main__":
    

    env_name = "gridmaze_29"

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    np.random.seed(0)

    env = gym.make(env_id, seed=42, no_goal=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, )

    rodc = RODCycle_DR(env, learn_rep_iteration=1, num_options=1, representation_step_size=0.03, dataset_size=100, p_option=0.1, n_steps=200, lambd=1.3)


    rewards, visit_percentage = rodc.rod_cycle(n_iterations=100)

    for i in range(1, len(rewards)):
        rewards[i] += rewards[i - 1]

    # save video
    os.chdir("minigrid_basics/DR_ROD")
    # for prefix in ['option', 'cumulative_visit', 'eigenvector']:
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', f'iteration%d.png', '-r', '30','-pix_fmt', 'yuv420p', 
        '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        '-y', f'ROD_{env_name}.mp4'
    ])

    for file_name in  glob.glob("*.png"):
        os.remove(file_name)
