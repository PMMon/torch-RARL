import os, sys
import argparse
import yaml
import warnings
import optuna
import time
import pickle as pkl
import gym
import numpy as np 
from collections import OrderedDict
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch

from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import BasePruner, MedianPruner, SuccessiveHalvingPruner
from optuna.visualization import plot_optimization_history, plot_param_importances

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize, VecTransposeImage, is_vecenv_wrapped
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.hyperparams_opt import HYPERPARAMS_SAMPLER
from utils.utils import get_latest_run_id, linear_schedule, get_wrapper_class, get_callback_list
from utils.callbacks import SaveVecNormalizeCallback, TrialEvalCallback, SetupProTrainingCallback, SetupAdvTrainingCallback
from utils.wrappers import AdversarialClassicControlWrapper, AdversarialMujocoWrapper
from models.algorithms import ALGOS


class ExperimentManager(object):
    """
    Reads and preprocesses hyperparameter. 
    Creates the environment and prepares the RL model for training.

    Taken and modified from RL Baselines3 Zoo 
    <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/exp_manager.py>

    MIT License

    Copyright (c) 2019 Antonin RAFFIN

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    """
    def __init__(self, 
        args: argparse.Namespace,
        algo: str,
        env_id: str,
        log_folder: str,
        tensorboard_log: str = "",
        n_timesteps: int = 0,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = -1,
        hyperparameter_path: str = "",
        hyperparams: Optional[Dict[str, Any]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        model_path: str = "",
        pretrained_model: str = "",
        optimize_hyperparameters: bool = False,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        n_opt_trials: int = 1,
        n_jobs: int = 1,
        sampler: str = "tpe",
        pruner: str = "median",
        optimization_log_path: Optional[str] = None,
        n_startup_trials: int = 0,
        n_evaluations_opt: int = 1,
        seed: int = 0,
        log_interval: int = 0,
        save_replay_buffer: bool = False,
        verbose: int = 1,
        vec_env_type: str = "dummy",
        n_envs: int = 1,
        n_eval_envs: int = 1,
        no_optim_plots: bool = False,
        adv_env: bool = False,
        adv_impact: str = "",
        adv_fraction: float = 1.0,
        adv_delay: int = -1,
        adv_index_list: List = None,
        adv_force_dim: int = 2,
        N_mu: int = 10, 
        N_nu: int = 10,
        device: str = None
    ) -> None:
        super(ExperimentManager, self).__init__()
        self.args = args
        self.seed = seed

        # algorithm
        self.algo = algo

        # environment
        self.n_envs = n_envs  # will be updated when reading hyperparams
        self.n_actions = None  # For DDPG/TD3 action noise objects
        self.env_id = env_id
        self.env_kwargs = {} if env_kwargs is None else env_kwargs
        self.normalize = False
        self.normalize_kwargs = {}
        self.env_wrapper = None
        self.frame_stack = None
        self.adv_env = adv_env

        self.vec_env_class = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_type]
        self.vec_env_kwargs = {}

        self._is_atari = self.is_atari(env_id)
        # self.vec_env_kwargs = {} if vec_env_type == "dummy" else {"start_method": "fork"}

        # adversarial env
        self.adv_fraction = adv_fraction
        self.adv_impact = adv_impact
        self.adv_index_list = adv_index_list
        self.adv_force_dim = adv_force_dim

        # training
        self.n_timesteps = n_timesteps
        self.save_freq = save_freq
        self.device = device

        # evaluation
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_envs = n_eval_envs
        self.eval_freq = eval_freq
        self.save_replay_buffer = save_replay_buffer

        # callbacks
        self.specified_callbacks = []
        self.callbacks = []

        # hyperparameters
        self.hyperparameter_path = hyperparameter_path
        self.custom_hyperparams = hyperparams
        self._hyperparams = {}
        
        # hyperparameter optimization config
        self.optimize_hyperparameters = optimize_hyperparameters
        self.optimize_hyperparameters_path = os.path.join(optimization_log_path, algo, env_id)
        self.storage = storage
        self.study_name = study_name
        self.no_optim_plots = no_optim_plots
    
        self.n_opt_trials = n_opt_trials    # maximum number of trials for finding the best hyperparams
        self.n_jobs = n_jobs     # number of parallel jobs when doing hyperparameter search
        self.sampler = sampler
        self.pruner = pruner
        self.n_startup_trials = n_startup_trials
        self.n_evaluations_opt = n_evaluations_opt
        self.deterministic_eval = not self.is_atari(self.env_id)

        # logging
        self.verbose = verbose
        self.tensorboard_log = None if tensorboard_log == "" else os.path.join(tensorboard_log, env_id)
        self.log_interval = log_interval

        # paths
        self.model_path = model_path
        if pretrained_model == "":
            self.save_path = os.path.join(self.model_path, f"{self.env_id}_{get_latest_run_id(self.model_path, self.env_id) + 1}")
            self.continue_training = False
        else: 
            self.save_path = os.path.join(self.model_path, pretrained_model)
            self.continue_training = True
        self.params_path = os.path.join(self.save_path, self.env_id)

        # RARL
        self.adv_delay = adv_delay
        self.N_mu = N_mu
        self.N_nu = N_nu


    def setup_experiment(self) -> Optional[BaseAlgorithm]:
        """Prepares experiment by creating environment, loading and preprocessing hyperparameters, etc.

        Returns:
            Optional[BaseAlgorithm]: return RL algorithm
        """
        hyperparams, sorted_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks = self._preprocess_hyperparams(hyperparams)

        # set up paths
        if not os.path.exists(self.params_path):
            os.makedirs(self.params_path)
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        if not os.path.exists(self.optimize_hyperparameters_path): 
            os.makedirs(self.optimize_hyperparameters_path)

        # create callbacks for train and test environments
        self.create_callbacks()

        # create environments
        n_envs = self.n_envs

        env = self.create_envs(n_envs, no_log=False)
        
        # preprocess action noise
        self._hyperparams = self._preprocess_action_noise(hyperparams, env)
        
        # define model and account for pre-trained model
        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, env)
        elif self.optimize_hyperparameters:
            return None
        else:
            # Train an agent from scratch
            model = ALGOS[self.algo](
                env=env,
                tensorboard_log=self.tensorboard_log,
                seed=self.seed,
                verbose=self.verbose,
                **self._hyperparams,
            )
        
        # save configs
        self._save_config(sorted_hyperparams)
        
        return model 


    def read_hyperparameters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Read hyperparameters from yaml file and order to for storage later

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: (parsed hyperparameters, ordered hyperparameters for storage)
        """
        # load hyperparameters from yaml file
        with open(os.path.join(self.hyperparameter_path, f"{self.algo}.yml"), "r") as f:
            hyperparams_dict = yaml.safe_load(f)
            
            # validate hyperparameter
            if self.env_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[self.env_id]
            elif self._is_atari:
                hyperparams = hyperparams_dict["atari"]
            else:
                raise ValueError(f"Hyperparameters not found for {self.algo}-{self.env_id}")


        if self.custom_hyperparams is not None:
            # overwrite hyperparams if needed
            hyperparams.update(self.custom_hyperparams)

        # sort hyperparams that will be saved
        sorted_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

        if self.verbose > 0:
            print("Default hyperparameters for environment (ones being tuned will be overridden):")
            pprint(sorted_hyperparams)

        return hyperparams, sorted_hyperparams
    

    def _preprocess_hyperparams(self, hyperparams: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Callable], List[BaseCallback]]:
        """Preprocess hyperparmeters

        Args:
            hyperparams (Dict[str, Any]): parsed hyperparameters

        Returns:
            Tuple[Dict[str, Any], Optional[Callable], List[BaseCallback]]: (hyperparameter that can be passed to model constructor,
                                                                                environment wrapper,
                                                                                list of callbacks
                                                                            )
        """
        self.n_envs = hyperparams.get("n_envs", 1)

        if self.verbose > 0:
            print(f"Using {self.n_envs} environments")

        # convert schedule strings to objects
        hyperparams = self._preprocess_schedules(hyperparams)

        # pre-process train_freq
        if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
            hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

        # overwrite number of timesteps
        if self.n_timesteps > 0:
            if self.verbose > 0:
                print(f"Overwriting n_timesteps with n={self.n_timesteps}")
        else:
            self.n_timesteps = int(hyperparams["n_timesteps"])
        
        # rarl - overwrite number of timesteps of protagonist and adversary
        if self.algo == "rarl":
            if self.N_mu > 0 and self.N_nu > 0:
                if self.verbose > 0:
                    print(f"Overwriting total_steps_protagonist with n_mu={self.N_mu}")
                    print(f"Overwriting total_steps_adversary with n_nu={self.N_nu}")
            else:
                self.N_mu = int(hyperparams["N_mu"])
                self.N_nu = int(hyperparams["N_nu"])

        if "adv_fraction" in hyperparams.keys(): 
            self.adv_fraction = hyperparams["adv_fraction"]
            del hyperparams["adv_fraction"]

        # pre-process normalization config
        hyperparams = self._preprocess_normalization(hyperparams)

        # pre-process policy/buffer keyword arguments
        for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs", "protagonist_kwargs", "protagonist_policy_kwargs", "adversary_kwargs", "adversary_policy_kwargs"}:
            if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
                hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

        # delete keys so the dict can be pass to the model constructor
        if "n_envs" in hyperparams.keys():
            del hyperparams["n_envs"]
        del hyperparams["n_timesteps"]

        if self.algo == "rarl":
            del hyperparams["N_mu"]
            del hyperparams["N_nu"]

        if "frame_stack" in hyperparams.keys():
            self.frame_stack = hyperparams["frame_stack"]
            del hyperparams["frame_stack"]

        # obtain a class object from a wrapper name string in hyperparams and delete the entry
        env_wrapper = get_wrapper_class(hyperparams)
        if "env_wrapper" in hyperparams.keys():
            del hyperparams["env_wrapper"]

        callbacks = get_callback_list(hyperparams)
        if "callback" in hyperparams.keys():
            self.specified_callbacks = hyperparams["callback"]
            del hyperparams["callback"]

        # manage devide
        if "device" in hyperparams.keys():
            self.device = hyperparams["device"]

        return hyperparams, env_wrapper, callbacks
    

    @staticmethod
    def _preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess schedules for learning

        Args:
            hyperparams (Dict[str, Any]): parsed hyperparameters

        Returns:
            Dict[str, Any]: hyperparameters with appropriate scheduler
        """
        # create schedules
        for key in ["learning_rate", "clip_range", "clip_range_vf", "delta_std"]:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split("_")
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constant_fn(float(hyperparams[key]))
            else:
                raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
        return hyperparams
    

    def _preprocess_normalization(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess normalization

        Args:
            hyperparams (Dict[str, Any]): parsed hyperparameters

        Returns:
            Dict[str, Any]: hyperparameters with appropriate normalization parameters
        """
        if "normalize" in hyperparams.keys():
            self.normalize = hyperparams["normalize"]

            # Special case, instead of both normalizing
            # both observation and reward, we can normalize one of the two.
            # in that case `hyperparams["normalize"]` is a string
            # that can be evaluated as python,
            # ex: "dict(norm_obs=False, norm_reward=True)"
            if isinstance(self.normalize, str):
                self.normalize_kwargs = eval(self.normalize)
                self.normalize = True

            # Use the same discount factor as for the algorithm
            if "gamma" in hyperparams:
                self.normalize_kwargs["gamma"] = hyperparams["gamma"]

            del hyperparams["normalize"]
        return hyperparams


    def create_callbacks(self):
        """Create callbacks for train and test environment
        """
        if self.save_freq > 0:
            # Account for the number of parallel environments
            self.save_freq = max(self.save_freq // self.n_envs, 1)
            self.callbacks.append(
                CheckpointCallback(
                    save_freq=self.save_freq,
                    save_path=self.save_path,
                    name_prefix="rl_model",
                    verbose=1,
                )
            )

        # Create test env if needed, do not normalize reward
        if self.eval_freq > 0 and not self.optimize_hyperparameters:
            # Account for the number of parallel environments
            self.eval_freq = max(self.eval_freq // self.n_envs, 1)

            if self.verbose > 0:
                print("Creating test environment...")

            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=self.params_path)
            eval_callback = EvalCallback(
                self.create_envs(self.n_eval_envs, eval_env=True),
                callback_on_new_best=save_vec_normalize,
                best_model_save_path=self.save_path,
                n_eval_episodes=self.n_eval_episodes,
                log_path=self.save_path,
                eval_freq=self.eval_freq,
                deterministic=self.deterministic_eval,
            )

            self.callbacks.append(eval_callback)


    def create_envs(self, n_envs: int, eval_env: bool = False, no_log: bool = False) -> VecEnv:
        """
        Create the environment and wrap it if necessary.
        :param n_envs: number of environments in stack
        :param eval_env: Whether is it an environment used for evaluation or not
        :param no_log: Do not log training when doing hyperparameter optim (issue with writing the same file)
        :return: the vectorized environment, with appropriate wrappers
        """
        # do not log eval env (issue with writing the same file)
        log_dir = None if eval_env or no_log else self.save_path

        monitor_kwargs = {}
        # special case for GoalEnvs: log success rate too
        if "Neck" in self.env_id or self.is_robotics_env(self.env_id) or "parking-v0" in self.env_id:
            monitor_kwargs = dict(info_keywords=("is_success",))

        # if adversarial environment, adapt action space by wrapping into adversarial wrapper
        wrapper_kwargs = {}

        if not eval_env:
            if self.algo == "rarl" or self.adv_env:
                #if not self.env_wrapper:
                if self.verbose > 0:
                    print("Using adversarial environment wrapper...")
                if self.adv_impact.lower() == "control":
                    self.env_wrapper = AdversarialClassicControlWrapper
                    wrapper_kwargs.update(dict(adv_fraction=self.adv_fraction))
                    if self.device:
                        wrapper_kwargs.update(dict(device=self.device))
                elif self.adv_impact.lower() == "force": 
                    self.env_wrapper = AdversarialMujocoWrapper
                    wrapper_kwargs.update(dict(adv_fraction=self.adv_fraction, index_list=self.adv_index_list, force_dim=self.adv_force_dim))
                    if self.device:
                        wrapper_kwargs.update(dict(device=self.device))

        # on most env, SubprocVecEnv does not help and is quite memory hungry, therefore we use DummyVecEnv by default
        env = make_vec_env(
            env_id=self.env_id,
            n_envs=n_envs,
            seed=self.seed,
            env_kwargs=self.env_kwargs,
            monitor_dir=log_dir,
            wrapper_class=self.env_wrapper,
            vec_env_cls=self.vec_env_class,
            vec_env_kwargs=self.vec_env_kwargs,
            monitor_kwargs=monitor_kwargs,
            wrapper_kwargs=wrapper_kwargs
        )

        # wrap the env into a VecNormalize wrapper if needed and load saved statistics when present
        env = self._maybe_normalize(env, eval_env)

        # optional frame-stacking
        if self.frame_stack is not None:
            n_stack = self.frame_stack
            env = VecFrameStack(env, n_stack)
            if self.verbose > 0:
                print(f"Stacking {n_stack} frames")

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, gym.spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (is_image_space(space) and not is_image_space_channels_first(space))
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space)

            if wrap_with_vectranspose:
                if self.verbose > 0:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env


    def _maybe_normalize(self, env: VecEnv, eval_env: bool) -> VecEnv:
        """
        Wrap the env into a VecNormalize wrapper if needed and load saved statistics when present.
        :param env: environment
        :param eval_env: True if evaluation mode, False otherwise
        :return: normalized environment
        """
        # pretrained model, load normalization
        path_ = os.path.dirname(self.save_path)
        path_ = os.path.join(path_, "vecnormalize.pkl")

        if os.path.exists(path_):
            print("Loading saved VecNormalize stats")
            env = VecNormalize.load(path_, env)
            # deactivate training and reward normalization while evaluating
            if eval_env:
                env.training = False
                env.norm_reward = False

        elif self.normalize:
            # copy to avoid changing default values by reference
            local_normalize_kwargs = self.normalize_kwargs.copy()
            # do not normalize reward for env used for evaluation
            if eval_env:
                if len(local_normalize_kwargs) > 0:
                    local_normalize_kwargs["norm_reward"] = False
                else:
                    local_normalize_kwargs = {"norm_reward": False}

            if self.verbose > 0:
                if len(local_normalize_kwargs) > 0:
                    print(f"Normalization activated: {local_normalize_kwargs}")
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **local_normalize_kwargs)
        return env


    def _preprocess_action_noise(self, hyperparams: Dict[str, Any], env: VecEnv) -> Dict[str, Any]:
        """Preprocesses action noise for exploration

        Args:
            hyperparams (Dict[str, Any]): parsed hyperparameter dict
            sorted_hyperparams (Dict[str, Any]): hyperparameters sorted
            env (VecEnv): environment

        Returns:
            Dict[str, Any]: tidied hyperparameter dict
        """
        # parse noise string - Note: only off-policy algorithms are supported
        if hyperparams.get("noise_type") is not None:
            noise_type = hyperparams["noise_type"].strip()
            noise_std = hyperparams["noise_std"]

            # save for later (hyperparameter optimization)
            self.n_actions = env.action_space.shape[0]

            if "normal" in noise_type:
                hyperparams["action_noise"] = NormalActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            elif "ornstein-uhlenbeck" in noise_type:
                hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            else:
                raise RuntimeError(f'Unknown noise type "{noise_type}"')

            print(f"Applying {noise_type} noise with std: {noise_std}")

            del hyperparams["noise_type"]
            del hyperparams["noise_std"]

        return hyperparams


    def _save_config(self, sorted_hyperparams: Dict[str, Any]) -> None:
        """
        Save unprocessed hyperparameters, this can be used to reproduce the experiment
        :param sorted_hyperparams: dict of sorted hyperparameters 
        """
        # Save hyperparams
        with open(os.path.join(self.params_path, "config.yml"), "w") as f:
            yaml.dump(sorted_hyperparams, f)

        # save command line arguments
        with open(os.path.join(self.params_path, "args.yml"), "w") as f:
            ordered_args = OrderedDict([(key, vars(self.args)[key]) for key in sorted(vars(self.args).keys())])
            yaml.dump(ordered_args, f)

        print(f"Save hyperparameters for reproducing results to: {self.params_path}")


    def learn(self, model: BaseAlgorithm) -> None:
        """ Trains given model
        :param model: an initialized RL model
        """
        kwargs = {}
        if self.log_interval > -1:
            kwargs = {"log_interval": self.log_interval}

        if len(self.callbacks) > 0:
            kwargs["callback"] = self.callbacks

            if self.algo == "rarl": 
                kwargs["callback_protagonist"] = self.callbacks
                callbacks_adversary = []
                kwargs["callback_adversary"] = [callbacks_adversary.append(callback) for callback in self.callbacks if not isinstance(callback, EvalCallback)]

        try:
            if self.algo == "rarl": 
                model.learn(self.n_timesteps, N_mu=self.N_mu, N_nu=self.N_nu, adv_delay=self.adv_delay, **kwargs)
            else:
                model.learn(self.n_timesteps, **kwargs)
        except KeyboardInterrupt:
            # this allows to save the model when interrupting training
            pass
        finally:
            # Release resources
            try:
                model.env.close()
            except EOFError:
                pass

    
    def _load_pretrained_agent(self, hyperparams: Dict[str, Any], env: VecEnv) -> BaseAlgorithm:
        # continue training
        print("Loading pretrained agent...")
        # policy should not be changed
        if self.algo == "rarl":
            del hyperparams["protagonist_policy"]
            del hyperparams["adversary_policy"]

            if "protagonist_policy_kwargs" in hyperparams.keys():
                del hyperparams["protagonist_policy_kwargs"]
                
            if "adversary_policy_kwargs" in hyperparams.keys():
                del hyperparams["adversary_policy_kwargs"]
        else:
            del hyperparams["policy"]

            if "policy_kwargs" in hyperparams.keys():
                del hyperparams["policy_kwargs"]


        model = ALGOS[self.algo].load(
            os.path.join(self.save_path, self.env_id),
            env=env,
            seed=self.seed,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
            **hyperparams,
        )

        replay_buffer_path = os.path.join(os.path.dirname(self.save_path), "replay_buffer.pkl")

        if os.path.exists(replay_buffer_path):
            print("Loading replay buffer...")
            model.load_replay_buffer(replay_buffer_path)
        return model


    def hyperparameters_optimization(self) -> None:
        """Optimize for hyperparameters
        """
        if self.verbose > 0:
            print("Optimizing hyperparameters")

        if self.storage is not None and self.study_name is None:
            warnings.warn(
                f"You passed a remote storage: {self.storage} but no `--study-name`."
                "The study name will be generated by Optuna, make sure to re-use the same study name "
                "when you want to do distributed hyperparameter optimization."
            )

        if self.tensorboard_log is not None:
            warnings.warn("Tensorboard log is deactivated when running hyperparameter optimization")
            self.tensorboard_log = None

        # TODO: eval each hyperparams several times to account for noisy evaluation
        sampler = self._create_sampler(self.sampler)
        pruner = self._create_pruner(self.pruner)

        if self.verbose > 0:
            print(f"Sampler: {self.sampler} - Pruner: {self.pruner}")

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True,
            direction="maximize",
        )

        try:
            study.optimize(self.objective, n_trials=self.n_opt_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        report_name = (
            f"report_{self.env_id}_{self.n_opt_trials}-trials-{self.n_timesteps}"
            f"-{self.sampler}-{self.pruner}_{int(time.time())}"
        )

        optimization_log_path = self.optimize_hyperparameters_path

        if self.verbose:
            print(f"Writing report to {optimization_log_path}")

        # Write report
        os.makedirs(os.path.dirname(optimization_log_path), exist_ok=True)
        study.trials_dataframe().to_csv(f"{optimization_log_path}.csv")

        # Save python object to inspect/re-use it later
        with open(f"{optimization_log_path}.pkl", "wb+") as f:
            pkl.dump(study, f)

        # Skip plots
        if self.no_optim_plots:
            return

        # Plot optimization result
        try:
            fig1 = plot_optimization_history(study)
            fig2 = plot_param_importances(study)

            fig1.savefig(os.path.join(optimization_log_path, "optimization_history.jpg"))
            fig2.savefig(os.path.join(optimization_log_path, "param_importances.jpg"))

        except (ValueError, ImportError, RuntimeError):
            pass


    def _create_sampler(self, sampler_method: str) -> BaseSampler:
        """Create sampler for hyperparameter optimization

        Args:
            sampler_method (str): sampler name

        Returns:
            BaseSampler: sampler
        """
        # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
        if sampler_method == "random":
            sampler = RandomSampler(seed=self.seed)
        elif sampler_method == "tpe":
            sampler = TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed, multivariate=True)
        elif sampler_method == "skopt":
            # cf https://scikit-optimize.github.io/#skopt.Optimizer
            # GP: gaussian process
            # Gradient boosted regression: GBRT
            sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
        else:
            raise ValueError(f"Unknown sampler: {sampler_method}")
        return sampler


    def _create_pruner(self, pruner_method: str) -> BasePruner:
        """Create pruner for hyperparameter optimization

        Args:
            pruner_method (str): pruner name

        Returns:
            BasePruner: pruner
        """
        if pruner_method == "halving":
            pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif pruner_method == "median":
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_evaluations_opt // 3)
        elif pruner_method == "none":
            # Do not prune
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_evaluations_opt)
        else:
            raise ValueError(f"Unknown pruner: {pruner_method}")
        return pruner


    def save_trained_model(self, model: BaseAlgorithm) -> None:
        """
        Save trained model optionally with its replay buffer and ``VecNormalize`` statistics
        :param model: trained model
        """
        print(f"Saving to {self.save_path}")
        model.save(os.path.join(self.save_path, self.env_id))

        if hasattr(model, "save_replay_buffer") and self.save_replay_buffer:
            print("Saving replay buffer")
            model.save_replay_buffer(os.path.join(self.save_path, "replay_buffer.pkl"))

        if self.normalize:
            # Important: save the running average, for testing the agent we need that normalization
            model.get_vec_normalize_env().save(os.path.join(self.params_path, "vecnormalize.pkl"))


    def objective(self, trial: optuna.Trial) -> float:
        """Objective for hyperparameter optimization

        Args:
            trial (optuna.Trial): optuna trial handler

        Returns:
            float: reward for config
        """
        kwargs = self._hyperparams.copy()

        # Hack to use DDPG/TD3 noise sampler
        trial.n_actions = self.n_actions

        # Sample candidate hyperparameters
        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](trial)

        if "adv_fraction" in sampled_hyperparams.keys(): 
            self.adv_fraction = sampled_hyperparams["adv_fraction"]
            del sampled_hyperparams["adv_fraction"]
        if "N_mu" in sampled_hyperparams.keys():
            self.N_mu = sampled_hyperparams["N_mu"]
            del sampled_hyperparams["N_mu"]
        if "N_nu" in sampled_hyperparams.keys():
            self.N_nu = sampled_hyperparams["N_nu"]
            del sampled_hyperparams["N_nu"]

        kwargs.update(sampled_hyperparams)

        # Create environment
        n_envs = self.n_envs
        env = self.create_envs(n_envs, no_log=True)

        # Define model
        model = ALGOS[self.algo](
            env=env,
            tensorboard_log=None,
            # We do not seed the trial
            seed=None,
            verbose=0,
            **kwargs,
        )

        # Create evaluation environment
        eval_env = self.create_envs(n_envs=self.n_eval_envs, eval_env=True)
        
        if self.algo == "rarl": 
            self.n_timesteps = int(500000 / (self.N_mu * model.protagonist.n_steps * model.protagonist.env.num_envs))
            optuna_eval_freq = int((self.n_timesteps * self.N_mu * model.protagonist.n_steps * model.protagonist.env.num_envs) / self.n_evaluations_opt)
        else:
            optuna_eval_freq = int(self.n_timesteps / self.n_evaluations_opt)
        
        print(f"n_timesteps: {self.n_timesteps}")
        print(f"N_mu: {self.N_mu}")
        print(f"N_nu: {self.N_nu}")
        print(f"protagonist n_steps: {model.protagonist.n_steps}")
        print(f"adversary n_steps: {model.adversary.n_steps}")
        print(f"eval_frequency: {optuna_eval_freq}")

        # account for parallel envs
        optuna_eval_freq = max(optuna_eval_freq // self.n_envs, 1)

        # use non-deterministic eval for Atari
        path = None
        if self.optimize_hyperparameters_path is not None:
            path = os.path.join(self.optimize_hyperparameters_path, f"trial_{str(trial.number)}")
        callbacks = get_callback_list({"callback": self.specified_callbacks})
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=path,
            log_path=path,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=optuna_eval_freq,
            deterministic=self.deterministic_eval,
        )
        callbacks.append(eval_callback)

        learn_kwargs = {}

        if self.algo == "rarl": 
            learn_kwargs["callback_protagonist"] = callbacks
            callbacks_adversary = []
            learn_kwargs["callback_adversary"] = [callbacks_adversary.append(callback) for callback in callbacks if not isinstance(callback, EvalCallback)]

        try:
            if self.algo == "rarl":
                model.learn(self.n_timesteps, N_mu=self.N_mu, N_nu=self.N_nu, adv_delay=self.adv_delay, **learn_kwargs)
            else:
                model.learn(self.n_timesteps, callback=callbacks, **learn_kwargs)
            # free memory
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            # sometimes, random hyperparams can generate NaN -> free memory
            model.env.close()
            eval_env.close()

            # prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned()

        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward


    @staticmethod
    def is_atari(env_id: str) -> bool:
        """Check whether environment is Atari environment

        Args:
            env_id (str): environment string description

        Returns:
            bool: True if environment is Atari, False otherwise
        """
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "AtariEnv" in str(entry_point)

    
    @staticmethod
    def is_robotics_env(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "gym.envs.robotics" in str(entry_point) or "panda_gym.envs" in str(entry_point)