import os, sys
import argparse
import pandas as pd
from typing import Dict, List
import difflib
import yaml
import gym
import numpy as np
import torch
from progress.bar import Bar 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from utils.wrappers import AdversarialClassicControlWrapper, AdversarialMujocoWrapper, ObsNoiseWrapper

from models.algorithms import ALGOS
from utils.utils import get_saved_hyperparams, StoreDict, get_wrapper_class

# =====================================
#   Evaluate and benchmark RL agents
# =====================================


class BenchmarkManager(object):
    """
    Evaluate trained RL models. 
    Aims for a fair comparison of RL models in the spirit of
    Henderson et al.: "Deep Reinforcement Learning that Matters". (2017).
    <https://arxiv.org/abs/1709.06560>
    """
    def __init__(self,
        n_timesteps: int,
        log_dir: str = os.path.join("logging", "benchmarking"),
        filename: str = "",
        algo: str = "ppo",
        model_dir: str = os.path.join("saved_models"),
        model_ids: List = [1, 1], 
        model_range: bool = True,
        load_best: bool = False, 
        load_checkpoint: int = None,
        deterministic: bool = True,
        env_id: str = "HalfCheetah-v3",
        with_mujoco: bool = False,
        n_envs: int = 1,
        norm_reward: bool = False, 
        with_obs_noise: bool = False, 
        with_adv_impact: bool = False, 
        adversary: str = os.path.join("saved_models"),
        with_var_oc: bool = False,
        oc_specifier: str = None,
        oc_value: float = 1.0,
        env_kwargs: Dict = None,
        render: bool = False,
        verbose: int = 0,
        n_seeds: int = 1,
        num_threads: int = 1,
        device: str = "cpu"
        ):
        self.n_timesteps = n_timesteps

        # general
        self.verbose = verbose
        self.n_seeds = n_seeds
        self.num_threads = num_threads
        self.device = device
        
        # logging results
        self.log_dir = log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.filename = filename 

        # model specifications
        self.algo = algo
        self.model_dir = model_dir 

        assert len(model_ids) >= 1, f"Length of model-ids must be > 0. Currently it is: {model_ids}"
        if model_range:
            assert len(model_ids) <= 2, f"If model-range set to True, length of model-ids must be 1 <= model_ids <= 2."
            if len(model_ids) == 1: 
                model_ids.append(model_ids[0])
            self.model_ids = np.arange(model_ids[0], model_ids[1]+1)
        else: 
            self.model_ids = model_ids
        
        self.load_best = load_best
        self.load_checkpoint = load_checkpoint

        self.deterministic = deterministic

        # environment specifications
        self.env_id = env_id
        registered_envs = set(gym.envs.registry.env_specs.keys())

        # if environment is not found, suggest the closest match
        if env_id not in registered_envs:
            try:
                closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
            except IndexError:
                closest_match = "'no close environment match found...'"
            raise ValueError(f"{env_id} not found in gym registry, did you maybe mean {closest_match}?")
        
        # off-policy algorithm only support one env
        off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "td3", "tqc"]

        if algo in off_policy_algos:
            self.n_envs = 1
        else: 
            self.n_envs = n_envs

        self.with_mujoco = with_mujoco
        self.norm_reward = norm_reward
        self.with_obs_noise = with_obs_noise
        self.with_adv_impact = with_adv_impact
        self.adversary = adversary
        self.with_var_oc = with_var_oc
        self.oc_specifier = oc_specifier
        self.oc_value = oc_value
        self.env_kwargs = env_kwargs

        self.is_atari = False
        self.is_bullet = False

        self.render = render

        # check if we are running python 3.8+, we need to patch saved model under python 3.6/3.7 to load them
        newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

        self.custom_objects = {}
        if newer_python_version:
            self.custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }


    def benchmark(self):
        results = {
            "algo": [],
            "env_id": [],
            "mean_reward": [],
            "std_reward": [],
            "n_timesteps": [],
            "eval_timesteps": [],
            "eval_episodes": [],
            "n_seeds": [],
            "with_obs_noise": [],
            "with_adv_impact": [],
            "with_var_oc": []
        }
        
        if self.with_var_oc:
            results.update(dict(oc_specifier=[], oc_value=[]))

        overall_rewards = []

        # iterate over models
        for model_id in self.model_ids:
            if self.verbose >= 0: 
                print(f"Model: {self.algo}_{model_id}")
            
            rewards = []
            lengths = []

            # iterate over seeds 
            with Bar("Seed", max=(self.n_seeds)) as bar:
                for seed in range(1, self.n_seeds+1):
                    set_random_seed(seed)

                    # set number of threads 
                    if args.num_threads > 0: 
                        if args.verbose == 1: 
                            print(f"Setting torch.num_threads to {self.num_threads}")
                        torch.set_num_threads(self.num_threads)

                    # get environment
                    env = self.create_eval_envs(seed, model_id)
                    # get model 
                    trained_model_path = self.get_trained_model_path(model_id)

                    model_kwargs = {}

                    if self.algo == "rarl" and self.load_best: 
                        model_kwargs.update(dict(protagonist_path="../best_model"))

                    model = ALGOS[self.algo].load(trained_model_path, env=env, custom_objects=self.custom_objects, seed=seed, device=self.device, **model_kwargs)

                    # eval env wrapper
                    # ToDo: if self.with_adv_impact: (wrapper)
                    # ToDo: if self.with_var_oc: (callback)
                    if self.with_obs_noise: 
                        env = ObsNoiseWrapper(env)

                    # evaluate
                    episode_rewards, episode_lengths = self.evaluate(self.n_timesteps, model, env)

                    rewards.append(episode_rewards)
                    lengths.append(episode_lengths)

                    bar.next()

                # note results for model  
                results["algo"].append(f"{self.algo}_{model_id}")
                results["env_id"].append(self.env_id)
                results["mean_reward"].append(np.mean(rewards))
                results["std_reward"].append(np.std(rewards))
                results["n_timesteps"].append(self.n_training_timesteps)
                results["eval_timesteps"].append(np.cumsum(episode_lengths)[-1])
                results["eval_episodes"].append(len(episode_rewards))
                results["n_seeds"].append(self.n_seeds)
                results["with_obs_noise"].append(self.with_obs_noise)
                results["with_adv_impact"].append(self.with_adv_impact)
                results["with_var_oc"].append(self.with_var_oc)

                if self.with_var_oc:
                    results["oc_specifier"].append(self.oc_specifier)
                    results["oc_value"].append(self.oc_value)

                overall_rewards.append(rewards)

                print("\n")
                print(results["eval_timesteps"][-1], "timesteps")
                print(results["eval_episodes"][-1], "Episodes")
                print(f"Mean reward over all {self.n_seeds} seeds: {np.mean(rewards):.2f} +- {np.std(rewards):.2f}")

        # create final statistics 
        results["algo"].append(f"{self.algo}_avg")
        results["env_id"].append(self.env_id)
        results["mean_reward"].append(np.mean(overall_rewards))
        results["std_reward"].append(np.std(overall_rewards))
        results["n_timesteps"].append(self.n_training_timesteps)
        results["eval_timesteps"].append(np.cumsum(episode_lengths)[-1])
        results["eval_episodes"].append(len(episode_rewards))
        results["n_seeds"].append(self.n_seeds)
        results["with_obs_noise"].append(self.with_obs_noise)
        results["with_adv_impact"].append(self.with_adv_impact)
        results["with_var_oc"].append(self.with_var_oc)
        
        if self.with_var_oc:
            results["oc_specifier"].append(self.oc_specifier)
            results["oc_value"].append(self.oc_value)

        # create DataFrame
        results_df = pd.DataFrame(results)
        # sort results
        results_df = results_df.sort_values(by=["algo", "env_id"])

        # dump as csv file:
        if self.filename != "":
            self.filename += "_"

        if self.with_obs_noise: 
            self.filename += "noise_"

        self.filename += f"{self.algo}_benchmark"

        if self.load_best: 
            self.filename += "_bestmodel"
        self.filename += ".csv"

        results_df.to_csv(os.path.join(self.log_dir, self.filename) ,sep="," , index=False)
        print(f'Saved results to {os.path.join(self.log_dir, f"{self.filename}{self.algo}_benchmark.csv")}')


    def get_trained_model_path(self, model_id: int) -> str:
        """
        Obtains path to trained model in self.model_dir with model_id.
        :params model_id: model identifier 
        :return: Path to trained model
        """
        # build folder structure
        foldername = os.path.join(f"{self.env_id}_{model_id}")
        trained_model_dir = os.path.join(self.model_dir, self.algo, self.env_id, foldername)

        assert os.path.isdir(trained_model_dir), f"The {trained_model_dir} folder was not found!"

        # obtain desired model in trained_model_dir
        found = False

        if self.load_best:
            if self.algo == "rarl": 
                model_path = os.path.join(trained_model_dir, f"{self.env_id}")
                found = os.path.isfile(os.path.join(trained_model_dir, "best_model.zip"))
            else:
                model_path = os.path.join(trained_model_dir, "best_model.zip")
                found = os.path.isfile(model_path)
        elif self.load_checkpoint is not None:
            model_path = os.path.join(trained_model_dir, f"rl_model_{self.load_checkpoint}_steps.zip")
            found = os.path.isfile(model_path)
        else:
            if self.algo == "rarl": 
                model_path = os.path.join(trained_model_dir, f"{self.env_id}")
                found = os.path.isfile(os.path.join(model_path, "protagonist.zip"))
            else: 
                model_path = os.path.join(trained_model_dir, f"{self.env_id}.zip")
                found = os.path.isfile(model_path)

        if not found:
            raise ValueError(f"No model found for {self.algo} on {self.env_id}, path: {model_path}")

        return model_path


    def create_eval_envs(self, seed: int, model_id: int) -> VecEnv:
        """Create evaluation environment with appropriate wrapper

        Args:
            seed (int): random number seed
            model_id (int): id of model

        Returns:
            VecEnv: Evaluation environment
        """
        # check type of enviroment - Atari
        entry_point = gym.envs.registry.env_specs[self.env_id].entry_point
        self.is_atari = "AtariEnv" in str(entry_point)
        self.is_bullet = "pybullet_envs" in str(entry_point)

        # build folder structure
        foldername = os.path.join(f"{self.env_id}_{model_id}")
        trained_model_dir = os.path.join(self.model_dir, self.algo, self.env_id, foldername)

        stats_path = os.path.join(trained_model_dir, self.env_id)
        hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=self.norm_reward, test_mode=True)

        if not stats_path:
            raise FileNotFoundError(f"No hyperparameters found at {os.path.join(trained_model_dir, self.env_id)}")

        # load env_kwargs if existing
        env_kwargs = {}
        args_path = os.path.join(stats_path, "args.yml")
        if os.path.isfile(args_path):
            with open(args_path, "r") as f:
                loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
                if loaded_args["env_kwargs"] is not None:
                    env_kwargs = loaded_args["env_kwargs"]

        # overwrite with command line arguments
        if self.env_kwargs is not None:
            env_kwargs.update(self.env_kwargs)

        log_dir = os.path.join(self.log_dir, f"{self.algo}_{model_id}", f"{self.env_id}_{seed}")

        # get environment wrapper
        env_wrapper = get_wrapper_class(hyperparams)

        if "env_wrapper" in hyperparams.keys():
            del hyperparams["env_wrapper"]


        # if adversarial environment, adapt action space by wrapping into adversarial wrapper
        if self.algo == "rarl" or self.with_adv_impact:
            wrapper_kwargs = {}
            #if not self.env_wrapper:
            if self.verbose > 0:
                print("Using adversarial environment wrapper...")
            if loaded_args["adv_impact"] == "control":
                env_wrapper = AdversarialClassicControlWrapper
                wrapper_kwargs.update(dict(adv_fraction=loaded_args["adv_fraction"]))
                if self.device:
                    wrapper_kwargs.update(dict(device=self.device))
            elif loaded_args["adv_impact"] == "force": 
                env_wrapper = AdversarialMujocoWrapper
                wrapper_kwargs.update(dict(adv_low=loaded_args["adv_low"], adv_high=loaded_args["adv_high"], index_list=loaded_args["adv_index_list"], force_dim=loaded_args["adv_force_dim"]))
                if self.device:
                    wrapper_kwargs.update(dict(device=self.device))
        else: 
            wrapper_kwargs = {}


        vec_env_kwargs = {}
        vec_env_cls = DummyVecEnv
        if self.n_envs > 1 or (self.is_bullet and self.render):
            # force SubprocVecEnv for Bullet env as Pybullet envs does not follow gym.render() interface
            vec_env_cls = SubprocVecEnv

        env = make_vec_env(
            self.env_id,
            n_envs=self.n_envs,
            monitor_dir=log_dir,
            seed=seed,
            wrapper_class=env_wrapper,
            env_kwargs=env_kwargs,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs=vec_env_kwargs,
            wrapper_kwargs=wrapper_kwargs
        )

        # load saved stats for normalizing input and rewards
        if stats_path is not None:
            if hyperparams["normalize"]:
                if self.verbose > 0:
                    print("Loading running average")
                    print(f"with params: {hyperparams['normalize_kwargs']}")
                path_ = os.path.join(stats_path, "vecnormalize.pkl")
                if os.path.exists(path_):
                    env = VecNormalize.load(path_, env)
                    # Deactivate training and reward normalization
                    env.training = False
                    env.norm_reward = False
                else:
                    raise ValueError(f"VecNormalize stats {path_} not found")

            n_stack = hyperparams.get("frame_stack", 0)
            if n_stack > 0:
                print(f"Stacking {n_stack} frames")
                env = VecFrameStack(env, n_stack)

        # training timesteps formating
        if self.algo == "rarl": 
            self.n_training_timesteps = hyperparams["n_timesteps"] * hyperparams["n_steps_protagonist"] * hyperparams["N_mu"]
            if self.n_training_timesteps < 1e6:
                self.n_training_timesteps = f"{int(self.n_training_timesteps / 1e3)}k"
            else:
                self.n_training_timesteps = f"{int(self.n_training_timesteps / 1e6)}M"
        elif "n_timesteps" in hyperparams.keys():
            if hyperparams["n_timesteps"] < 1e6:
                self.n_training_timesteps = f"{int(hyperparams['n_timesteps'] / 1e3)}k"
            else:
                self.n_training_timesteps = f"{int(hyperparams['n_timesteps'] / 1e6)}M"
        else: 
            raise ValueError("No parameter 'n_timesteps' in hyperparameters!")

        return env


    def evaluate(self, n_timesteps, model, env):
        """Evaluate model for n_timesteps on environment

        Args:
            n_timesteps (int): Number of evaluation timesteps
            model: RL agent
            env: Environment

        Returns:
            list, list: Episode rewards, Episode lengths
        """
        # reset environment
        obs = env.reset()
        state = None

        # deterministic by default except for atari games
        if self.is_atari or not self.deterministic: 
            deterministic = False
        else:
            deterministic = True

        episode_reward = 0.0
        episode_rewards, episode_lengths = [], []
        ep_len = 0

        successes = []

        try:
            for step in range(n_timesteps):
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                obs, reward, done, infos = env.step(action)
                if self.render:
                    env.render("human")

                episode_reward += reward[0]
                ep_len += 1

                if self.n_envs == 1:
                    # For atari the return reward is not the atari score,
                    # get it from the infos dict
                    if self.is_atari and infos is not None:
                        episode_infos = infos[0].get("episode")
                        if episode_infos is not None and self.verbose >= 1:
                            print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                            print("Atari Episode Length", episode_infos["l"])

                    if done and not self.is_atari:
                        # NOTE: for env using VecNormalize, the mean reward
                        # is a normalized reward when `--norm_reward` flag is passed
                        if self.verbose >= 1:
                            print(f"Episode Reward: {episode_reward:.2f}")
                            print("Episode Length", ep_len)
                        episode_rewards.append(episode_reward)
                        episode_lengths.append(ep_len)
                        episode_reward = 0.0
                        ep_len = 0
                        state = None

                    if done and infos[0].get("is_success") is not None:
                        if self.verbose >= 1:
                            print("Success?", infos[0].get("is_success", False))

                        if infos[0].get("is_success") is not None:
                            successes.append(infos[0].get("is_success", False))
                            episode_reward, ep_len = 0.0, 0
        except KeyboardInterrupt:
            pass


        if args.verbose > 0 and len(successes) > 0:
            print(f"Success rate: {100 * np.mean(successes):.2f}%")

        if args.verbose > 0 and len(episode_rewards) > 0:
            print(f"{len(episode_rewards)} Episodes")
            print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

        if args.verbose > 0 and len(episode_lengths) > 0:
            print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

        env.close()

        return episode_rewards, episode_lengths


if __name__ == "__main__": 
    # Parse arguments
    parser = argparse.ArgumentParser("Benchmarking trained RL models")

    # General Configs 
    parser.add_argument('--verbose', type=int, default=0, help="Verbose mode")
    parser.add_argument("--num-threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--with-mujoco",  default=False, action="store_true", help="Benchmark on MuJoCo environment")
    parser.add_argument("--render", default=False, action="store_true", help="Render the environment")
    parser.add_argument('--device', type=str, default="cpu", help="Specify device")

    # Configs about models
    parser.add_argument('--algo', type=str, default="ppo", choices=list(ALGOS.keys()), help="Algorithm to be evaluated")
    parser.add_argument('--model-dir', type=str, default=os.path.join("saved_models"), help="Path to saved models")
    parser.add_argument('--model-ids', type=int, nargs='+', default=[1, 1], help="Identifier of saved model names")
    parser.add_argument('--model-range', default=True, action="store_false", help="Specify whether model-list is range or not")
    parser.add_argument("--load-best",  default=False, action="store_true", help="Load best model")
    parser.add_argument("--load-checkpoint", type=int, default=None, help="Load checkpoint model with checkpoint number specified")
    parser.add_argument('--deterministic', default=True, action="store_false", help="Specify whether model predicts deterministically")

    # Configs about logging
    parser.add_argument('--log-dir', type=str, default=os.path.join("logging", "benchmarking"), help="Path to benchmark logging")
    parser.add_argument('--filename', type=str, default="", help="Name of benchmarking file")


    # Configs about benchmarking
    parser.add_argument("-n", "--n-timesteps", type= int, default=15000, help="Number of timesteps")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--type", type=str, default="avg", choices=["avg", "max"], help="Number of environments")

    # Config about evaluation environment
    parser.add_argument('--env-id', type=str, default="HalfCheetah-v3", help="Environment identifier")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--norm-reward", action="store_true", default=False, help="Normalize reward if applicable")
    parser.add_argument("--with-obs-noise",  default=False, action="store_true", help="Add Gaussian noise to observations")
    parser.add_argument("--with-adv-impact",  default=False, action="store_true", help="Add adversarial attacks to evaluation environment")
    parser.add_argument('--adversary', type=str, default=os.path.join("saved_models"), help="Path to adversary")
    parser.add_argument("--with-var-oc",  default=False, action="store_true", help="Manipulate operating condition")
    parser.add_argument('--oc-specifier', type=str, default="gravity", help="Name of manipulated operating condition")
    parser.add_argument('--oc-value', type=float, default=1.0, help="New value of manipulated operating condition")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, help="Additional environment kwargs")

    args = parser.parse_args()

    BManager = BenchmarkManager(
        n_timesteps=args.n_timesteps,
        log_dir=args.log_dir, 
        algo=args.algo,
        model_dir=args.model_dir,
        model_ids=args.model_ids, 
        model_range=args.model_range,
        load_best=args.load_best, 
        load_checkpoint=args.load_checkpoint,
        deterministic=args.deterministic,
        env_id=args.env_id,
        with_mujoco=args.with_mujoco,
        n_envs=args.n_envs,
        norm_reward=args.norm_reward, 
        with_obs_noise=args.with_obs_noise, 
        with_adv_impact=args.with_adv_impact, 
        adversary=args.adversary,
        with_var_oc=args.with_var_oc,
        oc_specifier=args.oc_specifier,
        oc_value=args.oc_value,
        env_kwargs=args.env_kwargs,
        render=args.render,
        verbose=args.verbose,
        n_seeds=args.n_seeds,
        num_threads=args.num_threads,
        )
    
    BManager.benchmark()