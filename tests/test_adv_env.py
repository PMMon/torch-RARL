import os, sys
import argparse
from gym import spaces
import numpy as np
import torch 
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder


from utils.wrappers import AdversarialMujocoWrapper
from utils.policies import ConstantPolicy 


# ===============================
#  Test Adversarial Environment
# ===============================


def test_adv_force_qualitatively(seed: int = None, render: bool = False, force: List = [0., 0.], index_list: List = ["torso"]):
    """Test qualitatively whether force on agent is applied correctly

    Args:
        seed (int, optional): seed. Defaults to None.
        render (bool, optional): whether or not to render environment. Defaults to False.
        force (List, optional): list of force components. Defaults to [0., 0.].
        index_list (List, optional): string identifier for force contact points. Defaults to ["torso"].
    """
    if seed:
        set_random_seed(seed)
    print("Testing adversarial force on MuJoCo environment...")
    # use MuJoCo environment
    env_id = "HalfCheetah-v3"

    # define adversarial environment wrapper
    env_wrapper = AdversarialMujocoWrapper
    wrapper_kwargs = {}

    # define arguments for wrapper - apply 2D force on torso
    adv_fraction = max(force)

    wrapper_kwargs.update(dict(adv_fraction=adv_fraction, index_list=index_list, force_dim=2))

    # build environment
    env = make_vec_env(
        env_id=env_id,
        n_envs=1,
        seed=seed,
        wrapper_class=env_wrapper,
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=DummyVecEnv
    )

    # define protagonist and adversary - specify protagonist to do nothing
    response_vector_protagonist = torch.tensor([[0., 0., 0., 0., 0., 0.]])
    pro_policy_action_space = spaces.Box(torch.min(response_vector_protagonist).item() * np.ones(response_vector_protagonist.shape, dtype=np.float32), torch.max(response_vector_protagonist).item() * np.ones(response_vector_protagonist.shape, dtype=np.float32))
    protagonist_policy = ConstantPolicy(action_space=pro_policy_action_space, observation_space=env.observation_space, response_vector=response_vector_protagonist)

    # adversary
    assert len(force) == 2*len(index_list), "2-dim force for each contact point in index-list has to be specified!"

    response_vector_adversary = torch.tensor([force]) / adv_fraction
    print(f"force tensor: {torch.tensor([force])}")
    adv_policy_action_space = spaces.Box(torch.min(response_vector_adversary).item() * np.ones(response_vector_adversary.shape, dtype=np.float32), torch.max(response_vector_adversary).item() * np.ones(response_vector_adversary.shape, dtype=np.float32))
    adversary_policy = ConstantPolicy(action_space=adv_policy_action_space, observation_space=env.observation_space, response_vector=response_vector_adversary)

    # set environment to proper state
    env.set_attr("operating_mode", "protagonist")
    env.set_attr("_pro_policy", protagonist_policy)
    env.set_attr("_adv_policy", adversary_policy)

    # apply forces to environment
    print("Simulate environment...")
    episodes = 100

    # record interaction to evaluate qualitatively
    video_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "videos")

    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=episodes,
    )

    obs = env.reset()
    try:
        for episode in range(episodes + 1):
            action, _ = protagonist_policy.predict(obs, deterministic=False)
            obs, rew, done, info = env.step(action)
            if render:
                env.render()
    except KeyboardInterrupt:
        pass

    env.close()

    print(f"Number of episodes: ", episodes)
    print("Test finished. Please analyze video produced.")


def test_adv_force(seed: int = None, render: bool = False):
    """Test whether force on agent is applied correctly

    Args:
        seed (int, optional): seed. Defaults to None.
        render (bool, optional): whether or not to render environment. Defaults to False.
    """
    if seed:
        set_random_seed(seed)
    print("Testing adversarial force on MuJoCo environment...")
    # use MuJoCo environment
    env_id = "HalfCheetah-v3"

    # define force contact points 
    index_list = ["torso", "bfoot", "ffoot"]

    # define adversarial environment wrapper
    env_wrapper = AdversarialMujocoWrapper
    wrapper_kwargs = {}

    # define arguments for wrapper - apply 2D force on torso
    adv_fraction = 100.
    wrapper_kwargs.update(dict(adv_fraction=adv_fraction, index_list=index_list, force_dim=2))

    # build environment
    env = make_vec_env(
        env_id=env_id,
        n_envs=1,
        seed=seed,
        wrapper_class=env_wrapper,
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=DummyVecEnv
    )

    # define protagonist and adversary - specify protagonist to do nothing
    response_vector_protagonist = torch.tensor([[0., 0., 0., 0., 0., 0.]])
    pro_policy_action_space = spaces.Box(torch.min(response_vector_protagonist).item() * np.ones(env.action_space.shape, dtype=np.float32), torch.max(response_vector_protagonist).item() * np.ones(env.action_space.shape, dtype=np.float32))
    protagonist_policy = ConstantPolicy(action_space=pro_policy_action_space, observation_space=env.observation_space, response_vector=response_vector_protagonist)

    # adversary
    # 1) push torso of agent forward (torso, bfoot, ffoot)
    print("Start push-test forwards...")
    force = [100., 0., 0., 0., 0., 0.]
    assert len(force) == 2*len(index_list), "2-dim force for each contact point in index-list has to be specified!"

    response_vector_adversary = torch.tensor([force]) / adv_fraction
    print(f"force tensor: {torch.tensor([force])}")
    adv_policy_action_space = spaces.Box(torch.min(response_vector_adversary).item() * np.ones(response_vector_adversary.shape, dtype=np.float32), torch.max(response_vector_adversary).item() * np.ones(response_vector_adversary.shape, dtype=np.float32))
    adversary_policy = ConstantPolicy(action_space=adv_policy_action_space, observation_space=env.observation_space, response_vector=response_vector_adversary)

    # set environment to proper state
    env.set_attr("operating_mode", "protagonist")
    env.set_attr("_pro_policy", protagonist_policy)
    env.set_attr("_adv_policy", adversary_policy)

    # apply forces to environment
    print("Simulate environment...")
    episodes = 100

    obs = env.reset()

    try:
        for episode in range(episodes + 1):
            action, _ = protagonist_policy.predict(obs, deterministic=False)
            obs, rew, done, info = env.step(action)
            if episode == 0: 
                old_x_pos = info[0]["x_position"]

            new_x_pos = info[0]["x_position"]

            if render:
                env.render()

            assert old_x_pos <= new_x_pos, f"Agent should be pushed towards positive x-axis! However, \n" \
                                            f"old x_position: {old_x_pos}, new x_position: {new_x_pos}"
            
            old_x_pos = new_x_pos
    except KeyboardInterrupt:
        pass

    env.close()

    print(f"Number of episodes: ", episodes)
    print("Push-Test forwards succesfull.")

    # 2) push feet of agent upwards (torso, bfoot, ffoot)
    print("Start lift-test upwards...")
    force = [0., 0., 0., 100., 0., 100.]
    assert len(force) == 2*len(index_list), "2-dim force for each contact point in index-list has to be specified!"

    response_vector_adversary = torch.tensor([force]) / adv_fraction
    print(f"force tensor: {torch.tensor([force])}")
    adv_policy_action_space = spaces.Box(torch.min(response_vector_adversary).item() * np.ones(response_vector_adversary.shape, dtype=np.float32), torch.max(response_vector_adversary).item() * np.ones(response_vector_adversary.shape, dtype=np.float32))
    adversary_policy = ConstantPolicy(action_space=adv_policy_action_space, observation_space=env.observation_space, response_vector=response_vector_adversary)

    # set environment to proper state
    env.set_attr("operating_mode", "protagonist")
    env.set_attr("_pro_policy", protagonist_policy)
    env.set_attr("_adv_policy", adversary_policy)

    # apply forces to environment
    print("Simulate environment...")
    episodes = 100

    obs = env.reset()

    try:
        for episode in range(episodes + 1):
            action, _ = protagonist_policy.predict(obs, deterministic=False)
            obs, rew, done, info = env.step(action)

            if episode == 0: 
                old_y_tip_pos = obs[0, 9]
            
            # allow for some time to rise
            if episode % 10 == 0:
                new_y_tip_pos = obs[0, 9]

            if render:
                env.render()

            assert old_y_tip_pos <= new_y_tip_pos, f"Agent should be lifted upwards along positive y-axis! However, \n" \
                                                    f"old y_tip_position: {old_y_tip_pos}, new y_tip_position: {new_y_tip_pos}\n"
            
            old_y_tip_pos = new_y_tip_pos

    except KeyboardInterrupt:
        pass

    env.close()

    print(f"Number of episodes: ", episodes)
    print("Lift-Test upwards succesfull.")


if __name__ == "__main__": 
    # Parse arguments
    parser = argparse.ArgumentParser("Test adversarial environment")

    # General Configs 
    parser.add_argument('--seed', type=int, default=None, help="Specify seed")
    parser.add_argument("--render",  default=False, action="store_true", help="Render environment")

    # Configs about force
    parser.add_argument('--force', nargs='+', type=float, default=[0., 0.], help="Specify adversarial force components")
    parser.add_argument('--index-list', nargs='+', type=str, default=["torso"], help='Contact point for adversarial forces (for Mujoco environments)')

    # Script Behavior
    parser.add_argument("--test-adv-force-qual",  default=False, action="store_true", help="Test negative reward wrapper")
    parser.add_argument("--test-adv-force",  default=False, action="store_true", help="Test negative reward wrapper")

    args = parser.parse_args()

    if args.test_adv_force_qual: 
        test_adv_force_qualitatively(render=args.render, seed=args.seed, force=args.force, index_list=args.index_list)
    if args.test_adv_force:
        test_adv_force(render=args.render, seed=args.seed)