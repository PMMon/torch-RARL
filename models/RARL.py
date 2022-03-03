import os, sys
import argparse
import gym
import time
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union, Tuple

from sb3_contrib import TRPO
from stable_baselines3.common import utils
from stable_baselines3.common.on_policy_algorithm import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.vec_env import VecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.wrappers import AdversarialWrapper


class RARL(BaseAlgorithm):
    def __init__(
        self, 
        protag_policy: Union[str, Type[ActorCriticPolicy]],
        adversary_policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        protag_layers: List,
        adversary_layers: List,
        n_iter: int = 2048,
        n_steps_protagonist: int = 2048,
        n_steps_adversary: int = 2048,
        protagonist_kwargs: Optional[Dict[str, Any]] = None, 
        adversary_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        create_eval_env: bool = False,
        verbose: int = 0,
        monitor_wrapper: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        support_multi_env: bool = False,
        tensorboard_log: Optional[str] = None,
        ):
        super(RARL, self).__init__(policy=None, 
                                    env=env,
                                    policy_base=None,
                                    learning_rate=None,
                                    tensorboard_log=tensorboard_log, 
                                    verbose=verbose, 
                                    support_multi_env=support_multi_env, 
                                    create_eval_env=create_eval_env,
                                    monitor_wrapper=monitor_wrapper,   
                                    seed=seed,
                                    supported_action_spaces=supported_action_spaces                                     
                                    )
        self.n_iter = n_iter

        # define protagonist
        if protagonist_kwargs is None: 
            protagonist_kwargs = dict(learning_rate=1e-3,
                                     )

        protagonist_policy_kwargs = dict(net_arch=protag_layers)
        self.protagonist = TRPO(policy=protag_policy, env=env, n_steps=n_steps_protagonist, policy_kwargs=protagonist_policy_kwargs, verbose=self.verbose, tensorboard_log=tensorboard_log, **protagonist_kwargs)

        # define adversary
        if adversary_kwargs is None: 
            adversary_kwargs = dict(learning_rate=1e-3,
                                    )
        if isinstance(env, VecEnv):
            [env.envs[i].set_action_space(env.envs[i].adv_action_space) for i in range(len(env.envs))]
        else: 
             env.set_action_space(env.adv_action_space)
        #adversary_policy = ActorCriticPolicy(observation_space=env.observation_space, action_space=env.adv_action_space, lr_schedule=linear_schedule(0.001))
        adversary_policy_kwargs = dict(net_arch=adversary_layers)
        self.adversary = TRPO(policy=adversary_policy, env=env, n_steps=n_steps_adversary, policy_kwargs=adversary_policy_kwargs, verbose=self.verbose, tensorboard_log=tensorboard_log, **adversary_kwargs)
    

    def _setup_model(self) -> None:
        """Create networks, buffer and optimizers."""
        print("Protagonist and Adversary will be set up when learning starts")
        pass 
    

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        print("Learning rates are schedule for protagonist and adversary individually")
        pass

    
    def _update_learning_rate(self, optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        print("Learning rates are updated for protagonist and adversary individually")
        pass


    def _excluded_save_params(self) -> List[str]:
        """
        ToDO: Adapt for RARL
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy",
            "device",
            "env",
            "eval_env",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]


    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """
        ToDO: Adapt for RARL
        Get the name of the torch variables that will be saved with
        PyTorch ``th.save``, ``th.load`` and ``state_dicts`` instead of the default
        pickling strategy. This is to handle device placement correctly.

        Names can point to specific variables under classes, e.g.
        "policy.optimizer" would point to ``optimizer`` object of ``self.policy``
        if this object.

        :return:
            List of Torch variables whose state dicts to save (e.g. th.nn.Modules),
            and list of other Torch variables to store with ``th.save``.
        """
        state_dicts = ["policy"]

        return state_dicts, []
    

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """
        self.start_time = time.time()

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = self._get_eval_env(eval_env)

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, eval_env, eval_freq, n_eval_episodes, log_path)

        return total_timesteps, callback


    def learn(self,
            total_timesteps: int,
            total_timesteps_protagonist: int,
            total_timesteps_adversary: int,
            adv_delay: int = -1,
            callback: MaybeCallback = None,
            callback_protagonist: MaybeCallback = None,
            callback_adversary: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "RARL",
            tb_log_name_protagonist: str = "PRO",
            tb_log_name_adversary: str = "ADV", 
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
        ):

        iteration_i = 0

        # setup learning 
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while iteration_i < total_timesteps:
            if iteration_i > 0:
                reset_num_timesteps = False
            # perform protagonist rollout and training
            print("train protagonist...")
            self.protagonist = self.protagonist.learn(total_timesteps_protagonist, 
                                                    callback=callback_protagonist, 
                                                    log_interval=log_interval, 
                                                    eval_env=eval_env,
                                                    eval_freq=eval_freq,
                                                    n_eval_episodes=n_eval_episodes,
                                                    tb_log_name=tb_log_name_protagonist,
                                                    eval_log_path=eval_log_path,
                                                    reset_num_timesteps=reset_num_timesteps,
                                                    )
            
            if iteration_i == 0 or iteration_i > adv_delay:
                # perform adversary rollout and training
                print("train adversary")
                self.adversary = self.adversary.learn(total_timesteps_adversary, 
                                                        callback=callback_adversary, 
                                                        log_interval=log_interval, 
                                                        eval_env=eval_env,
                                                        eval_freq=eval_freq,
                                                        n_eval_episodes=n_eval_episodes,
                                                        tb_log_name=tb_log_name_adversary,
                                                        eval_log_path=eval_log_path,
                                                        reset_num_timesteps=reset_num_timesteps
                                                        )
            
            self.num_timesteps = self.protagonist.num_timesteps
            iteration_i += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration_i % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration_i, exclude="tensorboard")
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

        # callback on end 
        callback.on_training_end()

        return self 


    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.protagonist.policy.predict(observation, state, episode_start, deterministic)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser("Robust Adversarial RL (RARL)")

    # General Configs 
    parser.add_argument('--verbose', type=int, default=0, help='Verbose mode')

    # Configs for environment 
    parser.add_argument('--env', type=str, default="BipedalWalker-v3", help='OpenAI gym environment name')

    # Configs for RARL
    parser.add_argument('--protag_policy', type=str, default="MlpPolicy", help='Policy of protagonist')
    parser.add_argument('--adversary_policy', type=str, default="MlpPolicy", help='Policy of adversary')
    parser.add_argument('--protag_layers', nargs='+', type=int, default=[100, 100, 100], help='Layer specification for actor')
    parser.add_argument('--adversary_layers', nargs='+', type=int, default=[100, 100, 100], help='Layer specification for critic')

    parser.add_argument('--n_iter', type=int, default=50, help='Number of iterations of alternating optimization')
    parser.add_argument('--n_steps_protagonist', type=int, default=2048, help='Number of steps to run for each environment per protagonist update')
    parser.add_argument('--n_steps_adversary', type=int, default=2048, help='Number of steps to run for each environment per adversary update')

    args = parser.parse_args()

    # Define model
    env = gym.make(args.env)
    env = AdversarialWrapper(env=env, adv_fraction=2.0)


    # Test model
    rarl_model = RARL(protag_policy="MlpPolicy", adversary_policy="MlpPolicy", env=env, protag_layers=args.protag_layers, adversary_layers=args.adversary_layers, tensorboard_log="tb_log", verbose=args.verbose)
    rarl_model.learn(10, 2049, 2049, adv_delay=args.adv_delay, eval_log_path="tb_log")