# Robust Adversarial Reinforcement Learning

The project at hand implements the Robust Adversarial Reinforcement Learning [(RARL)](https://arxiv.org/abs/1703.02702) agent, first introduced by Pinto et al. [[1]](#1). The code is based on [Stable Baslines3 (SB3)](https://github.com/DLR-RM/stable-baselines3) [[2]](#2) and [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) [[3]](#3).


## Setup
All code was developed and tested on Ubuntu 20.04 with Python 3.8.

To run the current code, we recommend to setup a virtual environment: 

```bash
python3 -m venv env                     # Create virtual environment
source env/bin/activate                 # Activate virtual environment
pip install -r requirements.txt         # Install dependencies
# Work for a while
deactivate                              # Deactivate virtual environment
```

Furthermore, [MuJoCo](https://mujoco.org/) needs to be installed. An installation guide can be found [here](https://github.com/openai/mujoco-py).

## Train the RARL agent

Similar to [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), the hyperparameters of all RL-agents are defined in `hyperparameters/algo_name.yml`. 

If the hyperparameters for a specific environment `env_id` are defined in the file, then the agent can be trained using: 

```
python scripts/train_adversary.py --algo rarl --env env_id
```

It is possible to specify the total number of iterations N<sub>iter</sub>, as well as the number of iterations for the protagonist N<sub>μ</sub> and adversary N<sub>ν</sub> using:

```
python scripts/train_adversary.py --algo rarl --env env_id --n_timesteps N_iter --total_steps_protagonist Nμ --total_steps_adversary Nν
```

A detailed explanation of all possible command-line flags can be found [here](ReadMeFiles/ARGUMENTS.md).

## Train other RL agents

Besides [RARL](https://arxiv.org/abs/1703.02702), a variety of other RL agents can be trained. A list of available algorithms can be found in the table below: 

| **Name**         | **Recurrent**      | `Box`          | `Discrete`     | `MultiDiscrete` | `MultiBinary`  | **Multi Processing**              |
| ------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| A2C<sup>[1](#f1)</sup>   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DDPG<sup>[1](#f1)</sup>  | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| DQN<sup>[1](#f1)</sup>   | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark: |
| PPO<sup>[1](#f1)</sup>   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| QR-DQN<sup>[2](#f2)</sup>  | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark: |
| SAC<sup>[1](#f1)</sup>   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| TD3<sup>[1](#f1)</sup>   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| TQC<sup>[2](#f2)</sup>   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x: | :heavy_check_mark: |
| TRPO<sup>[2](#f2)</sup>  | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| RARL<sup>[3](#f3)</sup>  | :x: | :heavy_check_mark: | :x: | :x:  | :x: | :x: |

<b id="f1">1</b>: Implemented in [SB3](https://github.com/DLR-RM/stable-baselines3) GitHub repository. 

<b id="f2">2</b>: Implemented in [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) GitHub repository.

<b id="f3">3</b>: Implemented by this GitHub repository.

To train a respective RL-agent, simply run the following code: 

```
python scripts/train_adversary.py --algo algo_name --env env_id
```

## References
<a id="1">[1]</a> Lerrel Pinto, James Davidson, Rahul Sukthankar, and Abhinav Gupta. “Robust Adversarial Reinforcement Learning.” In: arXiv:1703.02702 \[cs\] (Mar. 2017). arXiv: 1703.02702. URL: [http://arxiv.org/abs/1703.02702](http://arxiv.org/abs/1703.02702)

<a id="2">[2]</a> Antonin Raﬀin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. “Stable-Baselines3: Reliable Reinforcement Learning Implementations.” In: Journal of Machine Learning Research 22.268 (2021), pp. 1–8. URL: [http://jmlr.org/papers/v22/20-1364.html](http://jmlr.org/papers/v22/20-1364.html).

<a id="3">[3]</a> Antonin Raﬀin. RL Baselines3 Zoo. [https://github.com/DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo). 2020