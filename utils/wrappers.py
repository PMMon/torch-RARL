import gym
from gym import Env, spaces
import numpy as np
# ================================================
#   Define and customize gym environment wrapper
# ================================================


class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.

    Taken from RL Baselines3 Zoo 
    <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/wrappers.py>

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

    def __init__(self, env: Env, reward_offset: float = 0.0, n_successes: int = 1):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset
        self.n_successes = n_successes
        self.current_successes = 0

    def reset(self):
        self.current_successes = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info.get("is_success", False):
            self.current_successes += 1
        else:
            self.current_successes = 0
        # number of successes in a row
        done = done or self.current_successes >= self.n_successes
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


class AdversarialWrapper(gym.Wrapper):
    """
    Adapts the action space of the gym environment for the adversary
    """
    def __init__(self, env: Env, adv_fraction: float = 1.0):
        super(AdversarialWrapper, self).__init__(env)
        self.env = env
        # define adversarial impact
        adv_magnitude = env.action_space.high[0] * adv_fraction
        high_adv = np.ones(env.action_space.shape[0]) * adv_magnitude

        self._adv_action_space = self.convert_gym_space(env.action_space, low_val=-high_adv, high_val=high_adv)
        self._action_space = self.convert_gym_space(env.action_space, low_val=env.action_space.low, high_val=env.action_space.high)
        self._observation_space = self.convert_gym_space(env.observation_space, low_val=env.observation_space.low, high_val=env.observation_space.high)
        
    
    @property
    def observation_space(self):
        return self._observation_space


    @property
    def action_space(self):
        return self._action_space


    @property
    def adv_action_space(self):
        return self._adv_action_space


    def reset(self):
        return self.env.reset()


    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return next_obs, reward, done, info

    def render(self):
        self.env.render()


    def set_action_space(self, updated_action_space): 
        self._action_space = updated_action_space


    def set_action_space(self, updated_action_space): 
        self._action_space = updated_action_space


    def convert_gym_space(self, space, low_val, high_val):
        """Converts gym space into appropriate categories

        Args:
            space (gym.spaces): gym space

        Returns:
            gym.spaces: converted gym space
        """
        if isinstance(space, gym.spaces.Box):
            return spaces.box.Box(low=low_val, high=high_val)
        elif isinstance(space, gym.spaces.Discrete):
            return spaces.discrete.Discrete(n=space.n)
        elif isinstance(space, gym.spaces.Tuple):
            return spaces.prodcut.Product([self.convert_gym_space(x) for x in space.spaces])
        else:
            raise NotImplementedError


class AdversaryRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, rew):
        # modify rew
        return -rew

        