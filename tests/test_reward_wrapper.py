import os, sys
import argparse
import gym 
import numpy as np 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import safe_mean, set_random_seed
from stable_baselines3.common.base_class import maybe_make_env


from utils.wrappers import NegativeRewardWrapper, NegativeRewardVecEnvWrapper
from models.algorithms import ALGOS


# =====================
#  Test reward wrapper
# =====================

def test_negative_reward(algo, render=False, device="cpu", train_steps=20000, seed=2, vec_env = False):
    set_random_seed(seed)
    print("Testing negative reward wrapper...")
    # use Cartpole-v1 environment as it comes with only positive rewards (+1 for every step taken)
    environment_name = "CartPole-v1"

    if vec_env: 
        env = make_vec_env(env_id=environment_name, n_envs=1, seed=seed)
    else: 
        env = gym.make(environment_name)
        if seed is not None:
            env.seed(seed)
            env.action_space.seed(seed)
    
    # define model
    model = ALGOS[algo](policy="MlpPolicy", env=env, device=device, verbose=0, seed=seed)
    print(f"train model for {train_steps} steps...")
    model.learn(total_timesteps=train_steps)

    print("sample from environment after positive learning:")
    episodes = 5
    mean_score = 0

    for episode in range(1, episodes+1): 
        obs = env.reset()
        done = False
        score = 0

        while not done:
            if render:
                env.render()
            action, _ = model.predict(obs, deterministic=False) # change to model's prediction
            obs, reward, done, info = env.step(action)  # change to observations
            score += reward
        
        mean_score += score
        print(f"Episode: ", episode, " Score: ", score)

    mean_score /= episodes
    print(f"Number of episodes: ", episodes, " Mean score: ", mean_score)

    env.close()

    # wrap environment for modified reward function
    if isinstance(env, VecEnv):
        adv_env = NegativeRewardVecEnvWrapper(env)
    else: 
        adv_env = NegativeRewardWrapper(env)
        adv_env = maybe_make_env(adv_env, 0)
        adv_env = model._wrap_env(adv_env, 0, True)

    # train model
    model.env = adv_env
    print(f"train model for {train_steps} steps...")
    model.learn(total_timesteps=train_steps)

    if not isinstance(env, VecEnv):
        assert safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer]) == -safe_mean([ep_info["l"] for ep_info in model.ep_info_buffer]), \
                f'Model reward != Negative environment reward. Instead:\n' \
                f'Model reward: {safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer])}\n' \
                f'Environment reward: {safe_mean([ep_info["l"] for ep_info in model.ep_info_buffer])}'

    print("Check whether training reduces total reward...")
    print("sample from environment after negative training")
    mean_score_after_training = 0

    for episode in range(1, episodes+1): 
        obs = env.reset()
        done = False
        score = 0

        while not done:
            if render:
                env.render()
            action, _ = model.predict(obs, deterministic=False) # change to model's prediction
            obs, reward, done, info = env.step(action)  # change to observations
            score += reward
            
        mean_score_after_training += score
        print(f"Episode: ", episode, " Score: ", score)

    mean_score_after_training /= episodes
    print(f"Number of episodes: ", episodes, " Mean score after training: ", mean_score_after_training)
    
    env.close()

    assert mean_score_after_training <= mean_score, "Mean after training bigger than before training!"

    print("Test successfull.")


if __name__ == "__main__": 
    # Parse arguments
    parser = argparse.ArgumentParser("Test reward wrapper")

    # General Configs 
    parser.add_argument('--model', type=str, default="ppo", help="Specify model")
    parser.add_argument('--device', type=str, default="cuda", help="Specify device")
    parser.add_argument("--render",  default=False, action="store_true", help="Render environment")
    parser.add_argument("--vec-env",  default=False, action="store_true", help="Use vectorized environment")


    # Script Behavior
    parser.add_argument("--test-negative-reward",  default=False, action="store_true", help="Test negative reward wrapper")

    args = parser.parse_args()

    if args.test_negative_reward: 
        test_negative_reward(args.model, render=args.render, device=args.device, vec_env=args.vec_env)