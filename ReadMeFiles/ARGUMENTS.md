# Training - Command-line Flags

You can configure the behavior of the script by a variety of command-line flags:

## General Configurations
* `--verbose`: Set verbose mode (0: no output, 1: INFO). Choices: [0, 1]. Default is 0.
* `--seed`: Set seed for random number generator. Default is -1.
* `--num-exps`: Specify number of experiments. Default is 1.
* `--num-threads`: Specify number of threads for PyTorch. Default is -1.

## Configurations for environment
* `--env`: Name of OpenAi Gym environment. Default is `BipedalWalker-v3`.
* `--n-envs`: Number of environments for stack. Default is 1.
* `--env-kwargs`: Specify additional environment arguments in dictionary.
* `--vec-env-type`: Type of vector environment. Choices: [`dummy`, `subproc`]. Default is `dummy`.
* `--adv-env`: Specify whether to use adversarial Gym environment. Default is `False`. 

## Configurations for model
* `--algo`: Reinforcement Learning algorithm. Default is `ppo`. 
* `--saved-models-path`: Path to where models are being saved. Default is `saved_models`. 
* `--pretrained-model`: Path to a pretrained agent to continue training.
* `--save-replay-buffer`: Whether to save the replay buffer (when applicable). Default is `False`.

## Configurations for hyperparameters
* `--hyperparameter`: Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10).
* `--optimize-hyperparameters`: Whether to run hyperparameter search. Default is `False`. 
* `--hyperparameter-path`: Path to saved hyperparameters.
* `--storage`: Database storage path if distributed optimization should be used. Default is `None`. 
* `--sampler`: Sampler to use when optimizing hyperparameters with Optuna. Choices: [`random`, `tpe`, `skopt`]. Default is `tpe`. 
* `--pruner`:Pruner to use when optimizing hyperparameters with Optuna. Choices: [`halving`, `median`, `none`]. Default is `median`. 
* `--optimization-log-path`: Path to save the evaluation log. Default is `hyperparam_optimization`. 
* `--n-opt-trials`: Number of trials when optimizing hyperparameters with Optuna. Default is 10. 
* `--no-optim-plots`: Do not plot results when performing hyperparameter optimization.
* `--n-jobs`: Number of parallel jobs when optimizing hyperparameters with Optuna. Default is 1. 
* `--n-startup-trials`: Number of trials before using optuna sampler. Default is 10.
* `--n-evaluations-opt`: Training policy evaluated every n-timesteps during hyperparameter optimization. Default is 20. 

## Configurations for training
* `--n-timesteps`: Number of timesteps for training the RL agent. Default is -1.
* `--device`: Specify device. Default is `cpu`. 
* `--save-freq`: Save model every k steps (if negative, no checkpoint). Default is -1.  
* `--log-interval`: Log results every k steps (if -1, no change). Default is -1. 

## Configurations for evaluation
* `--eval-freq`: Evaluate the agent every e steps (if negative, no evaluation). Default is 10000.  
* `--n-eval-envs`: Number of evaluation environments. Default is 1. 
* `--n-eval-episodes`: Number of evaluation episodes. Default is 5.  

## Configurations for logging results
* `--tensorboard-log`: Tensorboard log dir. Default is `tb_logging`.  
* `--log-folder`: Log folder for e.g. benchmarking. Default is `logging`. 

## Configurations for RARL
* `--protagonist-policy`: Policy of protagonist. Default is `MlpPolicy`. 
* `--adversary-policy`: Policy of adversary. Default is `MlpPolicy`. 
* `--N-mu`: Number of protagonist iterations. Default is -1.
* `--N-nu`: Number of adversary iterations. Default is -1.

## Configurations for Adversarial Environment
* `--adv-impact`: Define how adversary impacts agent. Choices: [`control`, `force`]. If `control`, the adversary impacts the final action command by adding its action onto the protagonist's action command. If `force`, the adversary applies force on agent (only possible for MuJoCo envs). Default is `control`.  
* `--adv-fraction`: Scaling factor for adversarial action. Default is 1.0. 
* `--adv-delay`: Postpone optimization of adversary. Default is -1. 
* `--adv-index-list`: Contact point for adversarial forces (for Mujoco environments). Default is `torso`. 
* `--adv-force-dim`: Dimension of adversarial force vector per component. Default is 2.
