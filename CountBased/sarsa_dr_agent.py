from agent import Agent
import numpy as np

def get_bonus(dr, idx, transform):
    if transform == 'l1':
        expl_bonus = np.linalg.norm(dr[idx], ord = 1)
    elif transform == 'l2':
        expl_bonus = np.linalg.norm(dr[idx], ord = 2)
    elif transform == 'log_l1':
        expl_bonus = np.log(np.linalg.norm(dr[idx], ord = 1))
    else:
        expl_bonus = np.log(np.linalg.norm(dr[idx], ord = 2))
    return expl_bonus


class Sarsa_DR(Agent):

    def __init__(self, env, step_size, step_size_dr, gamma, gamma_dr, lambda_dr, epsilon, beta, transform = 'l1'):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = step_size
        self.epsilon = epsilon
        self.curr_s = self.env.get_current_state()

        # DR part
        self.beta = beta
        self.alpha_dr = step_size_dr
        self.gamma_dr = gamma_dr
        self.lambda_dr = lambda_dr
        self.num_states = self.env.get_num_states()
        self.num_acts = self.env.get_total_num_actions()
        self.dr = np.eye(self.num_states * self.num_acts)        
        self.transform = transform
        
        if self.transform not in ['l1', 'l2', 'log_l1', 'log_l2']:
            print("Defaulting to transform = log_l2...")
        

    def step(self):

        # DR step
        curr_a = self.epsilon_greedy(self.q[self.curr_s], epsilon=self.epsilon)
        env_r = self.env.act(curr_a)
        scaled_r = (env_r - self.env.max_reward) / self.env.max_reward
        next_s = self.env.get_current_state()
        next_a = self.epsilon_greedy(self.q[next_s], epsilon=self.epsilon)
        
        # DR step
        self.update_dr_values(self.curr_s, curr_a, next_s, next_a, scaled_r)
        
        # Estimate reward + bonus
        idx = self.curr_s * self.num_acts + curr_a
        expl_bonus = get_bonus(self.dr, idx, self.transform)
        actual_r = env_r + self.beta * expl_bonus
        
        # Sarsa step
        self.update_q_values(self.curr_s, curr_a, actual_r, next_s, next_a)
        
        self.curr_s = next_s

        self.current_undisc_return += env_r
        if self.env.is_terminal():
            self.episode_count += 1
            self.total_undisc_return += self.current_undisc_return
            self.current_undisc_return = 0

    def update_q_values(self, s, a, r, next_s, next_a):
        self.q[s][a] = self.q[s][a] + self.alpha * (r + self.gamma * (1.0 - self.env.is_terminal()) *
                                                    self.q[next_s][next_a] - self.q[s][a])

    def update_dr_values(self, s, a, next_s, next_a, r):
        
        idx = s * self.num_acts + a
        next_idx = next_s * self.num_acts + next_a
                
        for i in range(self.num_states * self.num_acts):
            cumulant = 1 if i == idx else 0
            target = np.exp(r / self.lambda_dr) * (cumulant + self.gamma_dr * (1.0 - self.env.is_terminal()) * self.dr[next_idx][i])
            self.dr[idx][i] = self.dr[idx][i] + self.alpha_dr * (target - self.dr[idx][i])