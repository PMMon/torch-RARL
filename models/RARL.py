import os, sys
import io
import pathlib
import argparse
import gym
import time
import torch
import numpy as np
from typing import Any, Dict, List, Iterable, Optional, Type, Union, Tuple

from stable_baselines3.common import utils
from stable_baselines3.common.on_policy_algorithm import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr, recursive_getattr, save_to_zip_file
from stable_baselines3.common.utils import get_system_info

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.wrappers import AdversaryRewardWrapper, AdversaryRewardVecEnvWrapper, AdversarialClassicControlWrapper
from utils.callbacks import SetupProTrainingCallback, SetupAdvTrainingCallback


class RARL(BaseAlgorithm):
    """
    Implements the Robust Adversarial Reinforcement Learning (RARL) agent introduced by
    Pinto et al. (2017) <https://arxiv.org/abs/1703.02702>. 

    An agent that operates in the presence of a destabilizing adversary which 
    applies disturbance forces to the system.
    """
    def __init__(
        self, 
        protagonist_policy: Union[str, Type[ActorCriticPolicy]],
        adversary_policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        n_steps_protagonist: int = 1024,
        n_steps_adversary: int = 1024,
        protagonist_kwargs: Optional[Dict[str, Any]] = None, 
        adversary_kwargs: Optional[Dict[str, Any]] = None,
        protagonist_policy_kwargs: Optional[Dict[str, Any]] = None, 
        adversary_policy_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        create_eval_env: bool = False,
        verbose: int = 0,
        monitor_wrapper: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        support_multi_env: bool = True,
        tensorboard_log: Optional[str] = None,
        device: Union[torch.device, str] = "auto",
        syn_timesteps: bool = False,
        protagonist_algo: str = "trpo",
        adversary_algo: str = "trpo",
        protagonist = None, 
        adversary = None
        ):
        super(RARL, self).__init__(policy=None, 
                                    env=env,
                                    policy_base=None,
                                    learning_rate=None,
                                    tensorboard_log=tensorboard_log, 
                                    verbose=verbose,
                                    device=device,
                                    support_multi_env=support_multi_env, 
                                    create_eval_env=create_eval_env,
                                    monitor_wrapper=monitor_wrapper,   
                                    seed=seed,
                                    supported_action_spaces=supported_action_spaces                                     
                                    )

        # general configs
        self.syn_timesteps = syn_timesteps
        self.device = device
        
        # define protagonist
        from models.algorithms import ALGOS_RARL
        self.protagonist_algo = protagonist_algo

        if protagonist: 
            self.protagonist = protagonist
        else: 
            if protagonist_kwargs is None:
                protagonist_kwargs = dict(
                                        batch_size= 128,
                                        gamma= 0.999,
                                        learning_rate= 0.007807660745967545,
                                        n_critic_updates= 5,
                                        cg_max_steps= 10,
                                        target_kl= 0.005,
                                        gae_lambda= 0.92,
                                        )

            self.protagonist_policy = protagonist_policy
            self.protagonist_kwargs = protagonist_kwargs
            self.protagonist_policy_kwargs = {} if protagonist_policy_kwargs is None else protagonist_policy_kwargs

            if "n_steps_protagonist" in self.protagonist_kwargs:
                n_steps_protagonist = self.protagonist_kwargs["n_steps_protagonist"]
                del self.protagonist_kwargs["n_steps_protagonist"]

            self.protagonist = ALGOS_RARL[self.protagonist_algo](policy=self.protagonist_policy, env=env, n_steps=n_steps_protagonist, policy_kwargs=self.protagonist_policy_kwargs, verbose=self.verbose, tensorboard_log=tensorboard_log, create_eval_env=create_eval_env, seed=seed, device=self.device, **self.protagonist_kwargs)

        # define adversary
        self.adversary_algo = adversary_algo

        if adversary: 
            self.adversary = adversary
        else: 
            if adversary_kwargs is None: 
                adversary_kwargs = dict(batch_size= 128,
                                        gamma= 0.999,
                                        learning_rate= 0.007807660745967545,
                                        n_critic_updates= 5,
                                        cg_max_steps= 10,
                                        target_kl= 0.005,
                                        gae_lambda= 0.92
                                        )

            self.adversary_policy = adversary_policy
            self.adversary_kwargs = adversary_kwargs
            self.adversary_policy_kwargs = {} if adversary_policy_kwargs is None else adversary_policy_kwargs

            if "n_steps_adversary" in self.adversary_kwargs: 
                n_steps_adversary = self.adversary_kwargs["n_steps_adversary"]
                del self.adversary_kwargs["n_steps_adversary"]

            # wrap environment for modified reward function
            if isinstance(env, VecEnv):
                adv_env = AdversaryRewardVecEnvWrapper(env)
            else: 
                adv_env = AdversaryRewardWrapper(env)

            self.adversary = ALGOS_RARL[self.adversary_algo](policy=self.adversary_policy, env=adv_env, n_steps=n_steps_adversary, policy_kwargs=self.adversary_policy_kwargs, verbose=self.verbose, tensorboard_log=tensorboard_log, create_eval_env=create_eval_env, seed=seed, device=self.device, **self.adversary_kwargs)

        
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
            pro_logger = utils.configure_logger(self.verbose, self.tensorboard_log, "PRO", reset_num_timesteps)
            adv_logger = utils.configure_logger(self.verbose, self.tensorboard_log, "ADV", reset_num_timesteps)

        self.protagonist.set_logger(pro_logger)
        self.adversary.set_logger(adv_logger)

        # Create eval callback if needed
        callback = self._init_callback(callback, eval_env, eval_freq, n_eval_episodes, log_path)

        return total_timesteps, callback


    def learn(self,
            total_timesteps: int,
            N_mu: int,
            N_nu: int,
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
            **kwargs
        ):
        """Setup learning configs and train model for total_timesteps

        Args:
            total_timesteps (int): N_iter, number of iterations for alternating RARL algorithm
            N_mu (int): number of iterations protagonist performs policy optimization
            N_nu (int): number of iterations adversary performs policy optimization
            adv_delay (int, optional): Number of iterations n until adversary is optimized. Defaults to -1.
            callback (MaybeCallback, optional): RARL callbacks. Defaults to None.
            callback_protagonist (MaybeCallback, optional): Protagonist callbacks. Defaults to None.
            callback_adversary (MaybeCallback, optional): Adversary callbacks. Defaults to None.
            log_interval (int, optional): Interval of logging. Defaults to 1.
            eval_env (Optional[GymEnv], optional): Evaluation environment. Defaults to None.
            eval_freq (int, optional): Evaluation frequency. Defaults to -1.
            n_eval_episodes (int, optional): Number of evaluation episodes. Defaults to 5.
            tb_log_name (str, optional): RARL log name. Defaults to "RARL".
            tb_log_name_protagonist (str, optional): Protagonist log name. Defaults to "PRO".
            tb_log_name_adversary (str, optional): Adversary log name. Defaults to "ADV".
            eval_log_path (Optional[str], optional): Path to evaluation logging. Defaults to None.
            reset_num_timesteps (bool, optional): Whether or not to reset number of current timesteps. Defaults to True.
        """
        iteration_i = 0

        # setup learning 
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        # protagonist and adversary callbacks
        if not callback_protagonist:
            callback_protagonist = []
        callback_protagonist.append(SetupProTrainingCallback(policy=self.adversary.policy,
                                                             verbose=self.verbose))
        if not callback_adversary:
            callback_adversary = []
        callback_adversary.append(SetupAdvTrainingCallback(policy=self.protagonist.policy,
                                                           verbose=self.verbose))

        while iteration_i < total_timesteps:
            if iteration_i > 0:
                if self.syn_timesteps:
                    self.protagonist.num_timesteps = max(self.protagonist.num_timesteps, self.adversary.num_timesteps - self.adversary.n_steps*self.adversary.env.num_envs)

            # perform protagonist rollout and training
            if self.verbose > 0:
                print("train protagonist...")
            
            # sync environment with protagonist
            if not reset_num_timesteps:
                self.protagonist._last_obs = np.array(self.protagonist.env.get_attr("_last_obs"), dtype=np.float32)
                self.protagonist._last_episode_starts = np.array(self.protagonist.env.get_attr("_last_episode_starts"), dtype=bool)

            if N_mu > 0:
                callback_protagonist[-1].policy = self.adversary.policy
                self.protagonist = self.protagonist.learn(N_mu*self.protagonist.n_steps*self.protagonist.env.num_envs, 
                                                        callback=callback_protagonist, 
                                                        log_interval=log_interval, 
                                                        eval_env=eval_env,
                                                        eval_freq=eval_freq,
                                                        n_eval_episodes=n_eval_episodes,
                                                        tb_log_name=tb_log_name_protagonist,
                                                        eval_log_path=eval_log_path,
                                                        reset_num_timesteps=reset_num_timesteps,
                                                        **kwargs
                                                        )
                self.num_timesteps = self.protagonist.num_timesteps
                reset_num_timesteps = False
            else:
                if iteration_i > 0:
                    reset_num_timesteps = False

            if iteration_i >= adv_delay:
                # perform adversary rollout and training
                if self.verbose > 0:
                    print("train adversary...")

                if self.syn_timesteps:
                    self.adversary.num_timesteps = max(self.adversary.num_timesteps, self.protagonist.num_timesteps - self.protagonist.n_steps*self.protagonist.env.num_envs)
                
                # sync environment with adversary
                self.adversary._last_obs = np.array(self.adversary.env.get_attr("_last_obs"), dtype=np.float32)
                self.adversary._last_episode_starts = np.array(self.adversary.env.get_attr("_last_episode_starts"), dtype=bool)

                if N_nu > 0:
                    callback_adversary[-1].policy = self.protagonist.policy
                    self.adversary = self.adversary.learn(N_nu*self.adversary.n_steps*self.adversary.env.num_envs, 
                                                            callback=callback_adversary, 
                                                            log_interval=log_interval, 
                                                            eval_env=eval_env,
                                                            eval_freq=eval_freq,
                                                            n_eval_episodes=n_eval_episodes,
                                                            tb_log_name=tb_log_name_adversary,
                                                            eval_log_path=eval_log_path,
                                                            reset_num_timesteps=reset_num_timesteps,
                                                            **kwargs
                                                            )
                    if N_mu <= 0: 
                        self.num_timesteps = self.adversary.num_timesteps

            if iteration_i % 10 == 0: 
                print(f"Iteration:", iteration_i, "/", total_timesteps)

            print(f"number of timesteps: {self.num_timesteps}")

            # update current progress
            iteration_i += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration_i % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration_i)
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


    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "protagonist",
            "adversary",
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
        state_dicts = []

        return state_dicts, []


    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
        exclude_pro: Optional[Iterable[str]] = None,
        include_pro: Optional[Iterable[str]] = None,
        exclude_adv: Optional[Iterable[str]] = None,
        include_adv: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        if self.verbose > 0: 
            print("saving model...")

        # save protagonist
        self.protagonist.save(os.path.join(path, "protagonist"), exclude=exclude_pro, include=include_pro)
        # save adversary
        self.adversary.save(os.path.join(path, "adversary"), exclude=exclude_adv, include=include_adv)

        # save own state_dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(os.path.join(path, "metadata"), data=data, params=params_to_save, pytorch_variables=pytorch_variables)


    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[torch.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        protagonist_path: str = "protagonist",
        adversary_path: str = "adversary",
        **kwargs,
    ) -> "BaseAlgorithm":
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters

        Taken and adapted from base model
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(os.path.join(path, "metadata"), device=device, custom_objects=custom_objects, print_system_info=print_system_info)

        # Remove stored device information and replace with ours
        if "protagonist_policy_kwargs" in data:
            if "device" in data["protagonist_policy_kwargs"]:
                del data["protagonist_policy_kwargs"]["device"]
        
        if "adversary_policy_kwargs" in data:
            if "device" in data["adversary_policy_kwargs"]:
                del data["adversary_policy_kwargs"]["device"]

        if "protagonist_policy_kwargs" in kwargs and kwargs["protagonist_policy_kwargs"] != data["protagonist_policy_kwargs"]:
            raise ValueError(
                f"The specified protagonist policy kwargs do not equal the stored protagonist policy kwargs."
                f"Stored kwargs: {data['protagonist_policy_kwargs']}, specified kwargs: {kwargs['protagonist_policy_kwargs']}"
            )
        
        if "adversary_policy_kwargs" in kwargs and kwargs["adversary_policy_kwargs"] != data["adversary_policy_kwargs"]:
            raise ValueError(
                f"The specified adversary policy kwargs do not equal the stored adversary policy kwargs."
                f"Stored kwargs: {data['adversary_policy_kwargs']}, specified kwargs: {kwargs['adversary_policy_kwargs']}"
            )

        # wrap environment for modified reward function
        if isinstance(env, VecEnv):
            adv_env = AdversaryRewardVecEnvWrapper(env)
        else: 
            adv_env = AdversaryRewardWrapper(env)

        # load submodels
        from models.algorithms import ALGOS_RARL
        protagonist = ALGOS_RARL[data["protagonist_algo"]].load(os.path.join(path, protagonist_path), env=env, print_system_info=False, device=device, custom_objects=custom_objects, force_reset=force_reset)
        adversary = ALGOS_RARL[data["adversary_algo"]].load(os.path.join(path, adversary_path), env=adv_env, print_system_info=False, device=device, custom_objects=custom_objects, force_reset=force_reset)

        # noinspection PyArgumentList
        model = cls(  # pytype: disable=not-instantiable,wrong-keyword-args
                protagonist_policy = data["protagonist_policy"],
                adversary_policy = data["adversary_policy"],
                env = env,
                device=device,  # pytype: disable=not-instantiable,wrong-keyword-args
            )

        # load parameters
        model.__dict__.update(dict(protagonist=protagonist, adversary=adversary))         
        model.__dict__.update(data)
        model.__dict__.update(kwargs)

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        return model


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser("Robust Adversarial RL (RARL)")

    # General Configs 
    parser.add_argument('--verbose', type=int, default=0, help='Verbose mode')
    parser.add_argument("--seed", type=int, default=-1, help="Random generator seed")

    # Configs for environment 
    parser.add_argument('--env', type=str, default="BipedalWalker-v3", help='OpenAI gym environment name')

    # Configs for RARL
    parser.add_argument('--protagonist_policy', type=str, default="MlpPolicy", help='Policy of protagonist')
    parser.add_argument('--adversary_policy', type=str, default="MlpPolicy", help='Policy of adversary')

    parser.add_argument('--n_iter', type=int, default=50, help='Number of iterations of alternating optimization')
    parser.add_argument('--N-mu', type=int, default=10, help='Number of iterations protagonist performs policy optimization')
    parser.add_argument('--N_nu', type=int, default=10, help='Number of iterations adversary performs policy optimization')
    parser.add_argument('--n_steps_protagonist', type=int, default=5000, help='Number of steps to run for each environment per protagonist update')
    parser.add_argument('--n_steps_adversary', type=int, default=5000, help='Number of steps to run for each environment per adversary update')

    parser.add_argument('--adv_delay', type=int, default=-1, help='Delay of adversary')
    parser.add_argument('--adv_fraction', type=float, default=1.0, help='Force-scaling for adversary')

    # Configs for learning
    parser.add_argument('--eval_freq', type=int, default=500, help='Frequency of evaluation')

    args = parser.parse_args()

    # Define model
    env = gym.make(args.env)
    env = AdversarialClassicControlWrapper(env=env, adv_fraction=args.adv_fraction)

    # Test model
    rarl_model = RARL(protagonist_policy=args.protagonist_policy, adversary_policy=args.adversary_policy, env=env, tensorboard_log="tb_log", verbose=args.verbose, seed=args.seed)
    rarl_model.learn(args.n_iter, N_mu=args.N_mu, N_nu=args.N_nu, adv_delay=args.adv_delay, eval_freq=args.eval_freq, eval_log_path="tb_log")
    
    # Save model
    rarl_model.save("RARL")