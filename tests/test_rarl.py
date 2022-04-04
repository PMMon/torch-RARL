import os, sys
import argparse
from gym import spaces
import numpy as np
import torch 
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize
from stable_baselines3.common.utils import safe_mean

from utils.wrappers import AdversarialMujocoWrapper
from utils.policies import ConstantPolicy 
from models.RARL import RARL


# ========================================
#  Test Robust Aversarial RL agent (RARL)
# ========================================


def test_silent_protagonist(train_steps: int = 10, seed: int = None, adv_fraction: float = 1.0, index_list: List = ["torso"], device: str = "cpu", verbose: int = 0, create_no_video: bool = True):
    if seed:
        set_random_seed(seed)
    print("Testing adversary given non-acting protagonist...")
    # use MuJoCo environment
    env_id = "HalfCheetah-v3"

    # define adversarial environment wrapper
    env_wrapper = AdversarialMujocoWrapper
    wrapper_kwargs = {}
    wrapper_kwargs.update(dict(adv_fraction=adv_fraction, index_list=index_list, force_dim=2))

    n_envs = 1

    # build environment
    env = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=env_wrapper,
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=DummyVecEnv
    )

    # define protagonist as non-acting unit
    response_vector_protagonist = torch.tensor([[0., 0., 0., 0., 0., 0.]])
    protagonist_policy = ConstantPolicy

    N_mu = 0
    protagonist_policy_kwargs = dict(response_vector=response_vector_protagonist)
    n_steps_protagonist = 2048
    n_steps_adversary = 2048

    if not create_no_video:
        # record interaction to evaluate qualitatively
        video_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "videos")

         # record beginning and ending of training
        env = VecVideoRecorder(
            env,
            video_folder,
            record_video_trigger=lambda x: x == train_steps - min(n_steps_protagonist*n_envs, 10000),
            video_length=min(train_steps*n_steps_protagonist*n_envs, 10000),
            name_prefix="pure_adversary"
        )

    model = RARL(
        protagonist_policy=protagonist_policy, 
        adversary_policy="MlpPolicy", 
        env=env, 
        seed=seed,
        verbose=verbose,
        n_steps_protagonist=n_steps_protagonist, 
        n_steps_adversary=n_steps_adversary,
        protagonist_policy_kwargs=protagonist_policy_kwargs,
        device=device)

    # train model
    print(f"train with N_iter = {train_steps}...")

    try:
        model.learn(train_steps, N_mu=N_mu, N_nu=1, adv_delay=0)
    except KeyboardInterrupt:
        # this allows to save the model when interrupting training
        model.env.close()
    
    model.env.close()

    # test whether reward is reasonably negative
    last_adv_reward = safe_mean([ep_info["r"] for ep_info in model.adversary.ep_info_buffer])
    assert last_adv_reward < -1e3, f"Adversary appears to not learn properly! Reward is {last_adv_reward}"

    print("Test finished successfully.")
    if not create_no_video:
        print("Please analyze video produced.")


def test_silent_adversary(train_steps: int = 100, seed: int = None, adv_fraction: float = 1.0, index_list: List = ["torso"], device: str = "cpu", verbose: int = 0, create_no_video: bool = False):
    if seed:
        set_random_seed(seed)
    print("Testing protagonist given non-acting adversary...")
    # use MuJoCo environment
    env_id = "HalfCheetah-v3"

    if train_steps < 100: 
        print("For proper testing, set train-steps to > 100!")
        print("Setting train-steps = 100.")
        train_steps = 100

    # define adversarial environment wrapper
    env_wrapper = AdversarialMujocoWrapper
    wrapper_kwargs = {}
    wrapper_kwargs.update(dict(adv_fraction=adv_fraction, index_list=index_list, force_dim=2))

    n_envs = 1

    # build environment
    env = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=env_wrapper,
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=DummyVecEnv
    )

    # define adversary as non-acting unit
    force = np.zeros(2*len(index_list))
    assert force.shape[0] == 2*len(index_list), "2-dim force for each contact point in index-list has to be specified!"

    response_vector_adversary = torch.tensor(np.expand_dims(force, axis=0)) / adv_fraction
    print(f"adversarial force tensor: {response_vector_adversary}")
    adversary_policy = ConstantPolicy

    N_nu = 0
    adversary_policy_kwargs = dict(response_vector=response_vector_adversary)
    n_steps_protagonist = 2048
    n_steps_adversary = 2048

    protagonist_kwargs = dict(
                            batch_size=128,
                            gamma=0.99,
                            learning_rate=1.9904955094948545e-05,
                            n_critic_updates=25,
                            cg_max_steps=30,
                            target_kl=0.01,
                            gae_lambda=0.95
                            )

    protagonist_policy_kwargs = dict(
                                    activation_fn=torch.nn.ReLU,
                                    net_arch=[dict(pi=[64, 64], vf=[64, 64])]
                                    )

    env = VecNormalize(env, norm_reward=False)

    if not create_no_video:
        # record interaction to evaluate qualitatively
        video_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "videos")

        # record beginning and ending of training
        env = VecVideoRecorder(
            env,
            video_folder,
            record_video_trigger=lambda x: x == train_steps*n_steps_protagonist*n_envs - min(n_steps_protagonist*n_envs, 10000),
            video_length=min(train_steps*n_steps_protagonist*n_envs, 10000),
            name_prefix="pure_protagonist"
        )

    model = RARL(
        protagonist_policy="MlpPolicy", 
        adversary_policy=adversary_policy, 
        env=env, 
        seed=seed,
        verbose=verbose,
        n_steps_protagonist=n_steps_protagonist, 
        n_steps_adversary=n_steps_adversary,
        protagonist_kwargs=protagonist_kwargs,
        protagonist_policy_kwargs=protagonist_policy_kwargs,
        adversary_policy_kwargs=adversary_policy_kwargs,
        device=device)

    # train model
    print(f"train with N_iter = {train_steps}...")

    try:
        model.learn(train_steps, N_mu=1, N_nu=N_nu, adv_delay=0)
    except KeyboardInterrupt:
        # this allows to save the model when interrupting training
        model.env.close()
    
    model.env.close()

    # test whether reward is reasonably negative
    last_pro_reward = safe_mean([ep_info["r"] for ep_info in model.protagonist.ep_info_buffer])
    assert last_pro_reward > 0, f"Protagonist appears to not learn properly! Reward is {last_pro_reward}"

    print("Test finished successfully.")
    if not create_no_video:
        print("Please analyze video produced.")

if __name__ == "__main__": 
    # Parse arguments
    parser = argparse.ArgumentParser("Test RARL")

    # General Configs 
    parser.add_argument('--seed', type=int, default=None, help="Specify seed")
    parser.add_argument('--verbose', type=int, default=0, help="Specify verbose level")
    parser.add_argument("--render",  default=False, action="store_true", help="Render environment")
    parser.add_argument('--device', type=str, default="cpu", help="Specify device")
    parser.add_argument("--create-no-video",  default=False, action="store_true", help="Create no video of training")

    # Configs about training
    parser.add_argument('--train-steps', type=int, default=10, help="Specify number of training iterations N_iter")

    # Configs about force
    parser.add_argument('--index-list', nargs='+', type=str, default=["torso"], help='Contact point for adversarial forces (for Mujoco environments)')
    parser.add_argument('--adv_fraction', type=float, default=1.0, help="Adversarial force scaling")

    # Script Behavior
    parser.add_argument("--test-silent-pro",  default=False, action="store_true", help="Test negative reward wrapper")
    parser.add_argument("--test-silent-adv",  default=False, action="store_true", help="Test negative reward wrapper")

    args = parser.parse_args()

    if args.test_silent_pro: 
        test_silent_protagonist(train_steps=args.train_steps, seed=args.seed, adv_fraction=args.adv_fraction, index_list=args.index_list, device=args.device, verbose=args.verbose, create_no_video=args.create_no_video)

    if args.test_silent_adv:
        test_silent_adversary(train_steps=args.train_steps, seed=args.seed, adv_fraction=args.adv_fraction, index_list=args.index_list, device=args.device, verbose=args.verbose, create_no_video=args.create_no_video)

