import os, sys
import argparse
import gym 
from gym import spaces
import numpy as np 
import torch 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from stable_baselines3.common.utils import set_random_seed


from utils.policies import ConstantPolicy

# ===============================
#  Test customized policies
# ===============================

def test_constant_policy(render=False, seed=None):
    """Test policy which samples constant vectors

    Args:
        render (bool, optional): whether or not to render environment. Defaults to False.
        seed (_type_, optional): seed. Defaults to None.
    """
    if seed:
        set_random_seed(seed)
    print("Testing constant policy...")
    # use continuous environment
    environment_name = "LunarLanderContinuous-v2"
    env = gym.make(environment_name)

    response_vector = torch.tensor([[100., -30.]])
    policy_action_space = spaces.Box(torch.min(response_vector).item() * np.ones(env.action_space.shape, dtype=np.float32), torch.max(response_vector).item() * np.ones(env.action_space.shape, dtype=np.float32))
    policy = ConstantPolicy(action_space=policy_action_space, observation_space=env.observation_space, respone_vector=response_vector)
    print(f"Response vector: {response_vector}")
    print("Sample from policy...")
    episodes = 5

    for episode in range(1, episodes+1): 
        obs = env.reset()
        done = False

        while not done:
            if render:
                env.render()
            action, _ = policy.predict(obs, deterministic=False)
            assert (action == response_vector[0, :].cpu().numpy()).all()
            obs, reward, done, info = env.step(action)

    print(f"Number of episodes: ", episodes)

    env.close()

    print("Test successfull.")


if __name__ == "__main__": 
    # Parse arguments
    parser = argparse.ArgumentParser("Test customized policies")

    # General Configs 
    parser.add_argument('--seed', type=int, default=None, help="Specify seed")
    parser.add_argument("--render",  default=False, action="store_true", help="Render environment")

    # Script Behavior
    parser.add_argument("--test-constant-policy",  default=False, action="store_true", help="Test negative reward wrapper")

    args = parser.parse_args()

    if args.test_constant_policy: 
        test_constant_policy(render=args.render, seed=args.seed)