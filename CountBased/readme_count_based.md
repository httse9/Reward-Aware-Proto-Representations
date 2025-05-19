# Reward-Aware Proto-Representations in Reinforcement Learning


## Count Based Exploration

To replicate the results in *Count-Based Exploration* section:

```
python3 tabular_sarsa_dr.py --input mdps/riverswim.mdp --num_episodes 100 --seed 7 --step_size 0.25 --step_size_dr 0.5 --lambda_dr 1 --epsilon 0.01 --beta 100 --transform log_l2
```

```
python3 tabular_sarsa_dr.py --input mdps/sixarms.mdp --num_episodes 100 --seed 7 --step_size 0.01 --step_size_dr 0.5 --lambda_dr 1.5 --epsilon 0.01 --beta 0.1 --imp_sampling 0 --transform log_l2
```
