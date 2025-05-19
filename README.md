# Reward-Aware Proto-Representations in Reinforcement Learning

This repository is the official implementation of Reward-Aware Proto-Representations in Reinforcement Learning. <!-- (https://arxiv.org/abs/2030.12345). -->

## Requirements

To install requirements:

```setup
conda create -n RAPR python=3.12
conda activate RAPR
pip install -r requirements.txt
```

## Training

### Reward Shaping

Example of reward shaping:
```reward shaping
python -m minigrid_basics.examples.run_reward_shaping --env fourrooms_2 --mode DR_potential --r_aux_weight 0.75 --step_size 1.0 --seed 1
```

For hyperparameters used to generate results in the paper, see minigrid_basics/hyperparameters/reward_shaping_exp.txt

To visualize:
```
python -m minigrid_basics.examples.figure4
```

### Option Discovery
Example of option discovery:
```option discovery
python -m minigrid_basics.examples.run_rod_cycle --env fourrooms_2 --representation DR --p_option 0.1 --dataset_size 100 --learn_rep_iteration 1 --representation_step_size 0.01 --num_options 1 --seed 1
```

Example of ROD cycle + Q-learning:
```rod+q
python -m minigrid_basics.examples.run_rod_cycle_q --env fourrooms_2 --representation DR --p_option 0.1 --dataset_size 100 --learn_rep_iteration 1 --representation_step_size 0.01 --num_options 1 --seed 1
```

For hyperparameters used to generate results in the paper, see minigrid_basics/hyperparameters/option-discovery_exp.txt

To visualize:
```
python -m  minigrid_basics.examples.rod_data_analysis
python -m  minigrid_basics.examples.rod_figure_34
python -m  minigrid_basics.examples.rod_figure_5
python -m  minigrid_basics.examples.rod_figure_7
```

### Count-Based Exploration
Refer to readme in CountBased

### Transfer
Example of transfer:
```transfer
python -m minigrid_basics.examples.run_default_features --seed 1
```

To visualize:
```
python -m minigrid_basics.examples.run_default_features --generate_test_data
python -m minigrid_basics.examples.default_features_plot
```

## Other visualization

To visualize env and eigenvectors:
```
# specify which figure to plot inside examples/figure_123.py, then
python -m minigrid_basics.examples.figure_123
```

## Results
See minigrid_basics/plots

## Contributing
Licensed under the Apache License, Version 2.0 (the "License")
