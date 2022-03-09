import os, sys
import argparse
import difflib
import gym
import numpy as np
import torch 

from stable_baselines3.common.utils import set_random_seed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from models.algorithms import ALGOS
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict

# ===========================================================================================
#   Training an RL agent on an environment specified. 
#   To promote standardization, the script is as close to RL Baselines3 Zoo as possible.
#   (https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/train.py)   
# ===========================================================================================

if __name__ == "__main__": 
    # Fetch CLI arguments
    parser = argparse.ArgumentParser("Robust Adversarial RL (RARL)")

    # General configs
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1], help="Verbose mode (0: no output, 1: INFO)")
    parser.add_argument("--seed", type=int, default=-1, help="Random generator seed")
    parser.add_argument("--num_threads", type=int, default=-1, help="Number of threads for PyTorch")

    # Configs about environment
    parser.add_argument('--env', type=str, default="BipedalWalker-v3", help='Name of gym environment')
    parser.add_argument("--n_envs", type=int, default=1, help="Number of environments for stack")
    parser.add_argument("--vec_env_type", type=str, default="dummy", choices=["dummy", "subproc"], help="VecEnv type")
    parser.add_argument("--env_kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("--adv_env", action="store_true", default=False, help="Adversarial gym environment")

    # Configs about model
    parser.add_argument("--algo", type=str, default="ppo", choices=list(ALGOS.keys()), help="RL Algorithm")
    parser.add_argument("--saved_models_path", type=str, default=os.path.join("saved_models"), help="Path to a saved models")
    parser.add_argument("--pretrained_model", type=str, default="", help="Path to a pretrained agent to continue training")
    parser.add_argument("--save_replay_buffer", default=False, action="store_true", help="Save the replay buffer (when applicable)")

    # Configs about hyperparameter
    parser.add_argument("-params", "--hyperparameter", type=str, nargs="+", action=StoreDict, help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)")
    parser.add_argument("-optimize", "--optimize_hyperparameters", action="store_true", default=False, help="Run hyperparameters search")
    parser.add_argument("--hyperparameter_path", type=str, default=os.path.join("hyperparameter"), help="Path to a saved models")
    parser.add_argument("--storage", type=str, default=None, help="Database storage path if distributed optimization should be used")
    parser.add_argument("--study_name", type=str, default=None, help="Study name for distributed optimization")
    parser.add_argument("--sampler", type=str, default="tpe", choices=["random", "tpe", "skopt"], help="Sampler to use when optimizing hyperparameters") 
    parser.add_argument("--pruner", type=str, default="median", choices=["halving", "median", "none"], help="Pruner to use when optimizing hyperparameters")
    parser.add_argument("--optimization_log_path", type=str, default=os.path.join("hyperparam_optimization"), help="Path to save the evaluation log and optimal policy for "
                                                                                                                        "each hyperparameter tried during optimization.")
    parser.add_argument("--n_opt_trials", type=int, default=10, help="Number of trials for optimizing hyperparameters.")
    parser.add_argument("--no_optim_plots", action="store_true", default=False, help="Disable hyperparameter optimization plots")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs when optimizing hyperparameters")
    parser.add_argument("--n_startup_trials", type=int, default=10, help="Number of trials before using optuna sampler")

    # Configs about training
    parser.add_argument("-n", "--n_timesteps", type=int, default=-1, help="Number of timesteps")
    parser.add_argument("--save_freq", type=int, default=-1, help="Save model every k steps (if negative, no checkpoint)")
    parser.add_argument("--log_interval", type=int, default=-1, help="Override log interval (default: -1, no change)")

    # Configs about evaluation
    parser.add_argument("--n_evaluations_opt", type=int, default=20, help="Training policies are evaluated every n-timesteps during hyperparameter optimization")
    parser.add_argument("--n_eval_episodes", type=int, default=5, help="Number of episodes to use for evaluation")
    parser.add_argument("--n_eval_envs", type=int, default=1, help="Number of environments for evaluation")
    parser.add_argument("--eval_freq", type=int, default=10000, help="Evaluate the agent every e steps (if negative, no evaluation).")

    # Configs about logging results 
    parser.add_argument("-tb", "--tensorboard_log", type=str, default=os.path.join("tb_logging"), help="Tensorboard log dir")
    parser.add_argument("-f", "--log_folder", type=str, default=os.path.join("logging"), help="Log folder")

    # Configs for RARL
    parser.add_argument('--protagonist_policy', type=str, default="MlpPolicy", help='Policy of protagonist')
    parser.add_argument('--adversary_policy', type=str, default="MlpPolicy", help='Policy of adversary')
    parser.add_argument('--protag_layers', nargs='+', type=int, default=[100, 100, 100], help='Layer specification for actor')
    parser.add_argument('--adversary_layers', nargs='+', type=int, default=[100, 100, 100], help='Layer specification for critic')

    parser.add_argument('--total_steps_protagonist', type=int, default=10, help='Number of steps to run for each environment per protagonist update')
    parser.add_argument('--total_steps_adversary', type=int, default=10, help='Number of steps to run for each environment per adversary update')

    parser.add_argument('--adv_delay', type=int, default=-1, help='Delay of adversary')
    parser.add_argument('--adv_fraction', type=float, default=1.0, help='Force-scaling for adversary')

    args = parser.parse_args()

    # check gym environment
    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())

    # if environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close environment match found...'"
        raise ValueError(f"{env_id} not found in gym registry, did you maybe mean {closest_match}?")
    
    # set random seed
    if args.seed < 0:
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item()
    
    set_random_seed(args.seed)

    # set number of threads 
    if args.num_threads > 0: 
        if args.verbose == 1: 
            print(f"Setting torch.num_threads to {args.num_threads}")
        torch.set_num_threads(args.num_threads)
    
    model_path = os.path.join(args.saved_models_path, args.algo.lower(), env_id)

    # check path
    if args.pretrained_model != "":
        assert os.path.isfile(os.path.join(model_path, args.pretrained_model, env_id + ".zip")), f"The pretrained_agent must be a valid path to a .zip file. Check path to {os.path.join(model_path, args.pretrained_model)}"
    
    # print information
    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {args.seed}")

    # experiment manager
    exp_manager = ExperimentManager(
        args, 
        algo=args.algo,
        env_id=env_id,
        log_folder=args.log_folder,
        tensorboard_log=args.tensorboard_log,
        n_timesteps=args.n_timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_freq=args.save_freq,
        hyperparameter_path=args.hyperparameter_path,
        hyperparams=args.hyperparameter,
        env_kwargs=args.env_kwargs,
        model_path=model_path,
        pretrained_model = args.pretrained_model,
        optimize_hyperparameters=args.optimize_hyperparameters,
        storage=args.storage,
        study_name=args.study_name,
        n_opt_trials=args.n_opt_trials,
        n_jobs=args.n_jobs,
        sampler=args.sampler,
        pruner=args.pruner,
        optimization_log_path=args.optimization_log_path,
        n_startup_trials=args.n_startup_trials,
        n_evaluations_opt=args.n_evaluations_opt,
        seed=args.seed,
        log_interval=args.log_interval,
        save_replay_buffer=args.save_replay_buffer,
        verbose=args.verbose,
        vec_env_type=args.vec_env_type,
        n_envs=args.n_envs,
        n_eval_envs=args.n_eval_envs,
        no_optim_plots=args.no_optim_plots,
        adv_env=args.adv_env, 
        adv_fraction=args.adv_fraction,
        adv_delay=args.adv_delay,
        total_steps_protagonist=args.total_steps_protagonist,
        total_steps_adversary=args.total_steps_adversary
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    model = exp_manager.setup_experiment()

    # Normal training
    if model is not None:
        exp_manager.learn(model)
        exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()