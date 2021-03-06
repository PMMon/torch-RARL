import gym
from gym import Env, spaces
import numpy as np
import torch
from typing import List

from stable_baselines3.common.utils import obs_as_tensor, get_device
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper, VecEnvStepReturn

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


class NegativeRewardWrapper(gym.RewardWrapper):
    """
    Negates the reward function
    """
    def __init__(self, env):
        super().__init__(env=env)


    def reward(self, rew):
        # modify rew
        return -rew


class NegativeRewardVecEnvWrapper(VecEnvWrapper):
    """
    Negates the reward function for vector environments
    """
    def __init__(self, venv: VecEnv):
        super().__init__(venv=venv)


    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs, self.reward(reward), done, info
    
    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def reset(self) -> np.ndarray:
        return self.venv.reset()
    
    def reward(self, rew):
        # modify rew
        return -rew


class AdversaryRewardWrapper(gym.RewardWrapper):
    """
    Adapts the reward function of the adversary in RARL
    """
    def __init__(self, env):
        self.action_space(env.__getattribute__("adv_action_space"))
        super().__init__(env=env)


    def reward(self, rew):
        # modify rew
        return -rew


class AdversaryRewardVecEnvWrapper(VecEnvWrapper):
    """
    Adapts the reward function of the adversary in RARL for vector environments
    """
    def __init__(self, venv: VecEnv):
        super().__init__(venv=venv, action_space=venv.get_attr("adv_action_space")[0])


    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs, -reward, done, info
    
    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def reset(self) -> np.ndarray:
        return self.venv.reset()


class AdversarialClassicControlWrapper(gym.Wrapper):
    """
    Adapts the action space of the gym environment for the adversary and couples 
    actions from protagonist and adversary during training
    """
    def __init__(self, env: Env, adv_fraction: float = 1.0, device: str = "auto"):
        super(AdversarialClassicControlWrapper, self).__init__(env)
        self.env = env
        # define adversarial impact
        adv_magnitude = env.action_space.high[0] * adv_fraction
        high_adv = np.ones(env.action_space.shape[0], dtype=np.float32) * adv_magnitude

        self._adv_action_space = self.convert_gym_space(env.action_space, low_val=-high_adv, high_val=high_adv)
        self._action_space = env.action_space
        self._initial_action_space = env.action_space
        self._observation_space = env.observation_space

        self.operating_mode = None
        self._pro_policy = None
        self._adv_policy = None

        self.device = get_device(device)

        self._last_obs = None
        self._last_episode_starts = None


    @property
    def observation_space(self):
        return self._observation_space


    @property
    def action_space(self):
        return self._action_space
    

    @property
    def initial_action_space(self):
        return self._initial_action_space


    @property
    def adv_action_space(self):
        return self._adv_action_space


    def set_action_space(self, updated_action_space): 
        self._action_space = updated_action_space


    def sample_action(self):
        class pro_adv_action(object):
            def __init__(self, pro_action, adv_action):
                self.pro_action = pro_action
                self.adv_action = adv_action

        return pro_adv_action(self.action_space.sample(), self.adv_action_space.sample())


    def step(self, action):
        if hasattr(action, '__dict__'):
            print("action has dict")
            obs, rew, done, info = self.env.step(action.pro_action + action.adv_action)
            self._last_obs = obs
            self._last_episode_starts = done
            return obs, rew, done, info
        else:
            if not self.operating_mode:
                obs, rew, done, info = self.env.step(action)
                self._last_obs = obs
                self._last_episode_starts = done
                return obs, rew, done, info

            elif self.operating_mode.lower() == "protagonist":
                with torch.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    if len(obs_tensor.shape) == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    action_sampled  = self._adv_policy._predict(obs_tensor, deterministic=True)

                clipped_actions = action_sampled.cpu().numpy()

                # Clip the actions to avoid out of bound error
                if isinstance(self._adv_action_space, gym.spaces.Box):
                    clipped_actions = np.clip(clipped_actions, self._adv_action_space.low, self._adv_action_space.high).squeeze(0)
                
            elif self.operating_mode.lower() == "adversary":
                with torch.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    if len(obs_tensor.shape) == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    action_sampled = self._pro_policy._predict(obs_tensor, deterministic=True)

                clipped_actions = action_sampled.cpu().numpy()

                # Clip the actions to avoid out of bound error
                if isinstance(self._action_space, gym.spaces.Box):
                    clipped_actions = np.clip(clipped_actions, self._action_space.low, self._action_space.high).squeeze(0)
                
            else:
                raise ValueError(f"Please choose operating mode either 'protagonist' or 'adversary', not: ", self.operating_mode)

            obs, rew, done, info = self.env.step(action + clipped_actions)
            self._last_obs = obs
            self._last_episode_starts = done

            return obs, rew, done, info
    

    def reset(self):
        self._last_obs = self.env.reset()
        return self._last_obs


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


class AdversarialMujocoWrapper(gym.Wrapper):
    """
    Modeling errors can be viewed as extra forces in the system.
    This wrapper allows the adversary to apply disturbing forces to the system in order
    to counteract the protagonist's goal.
    The wrapper replaces the customized environments defined in: 
    Lerrel Pinto: "Gym environments with adversarial disturbance agents" <https://github.com/lerrel/gym-adv>.
    """
    def __init__(self, env: Env, adv_fraction: float = 1.0, adv_low: float = 1.0, adv_high: float = 1.0, index_list: List[str] = [], force_dim: int = 2, device: str = "auto"):
        super(AdversarialMujocoWrapper, self).__init__(env)
        self.env = env
        self.force_dim = force_dim
        self.adv_fraction = adv_fraction

        # define point of attack
        self._adv_force_bname = index_list
        bnames = self.env.model.__getattribute__("body_names") 

        try: 
            self._adv_body_indicees = [bnames.index(i) for i in self._adv_force_bname]
        except: 
            raise AttributeError("Environment: %s does not include all body names in list: %s\n Please use body names available from here: %s" % (env.spec.id, index_list, bnames))
        
        adv_action_space_dim = force_dim*len(self._adv_body_indicees)
        # ToDo: Normalize action space and multiply by fraction in step to enable "superpowers"
        self._adv_action_space = spaces.Box(-1 * np.ones(adv_action_space_dim, dtype=np.float32), np.ones(adv_action_space_dim, dtype=np.float32))
        self._action_space = env.action_space
        self._initial_action_space = env.action_space
        self._observation_space = env.observation_space

        self.operating_mode = None
        self._pro_policy = None
        self._adv_policy = None

        self.device = get_device(device)

        self._last_obs = None
        self._last_episode_starts = None


    def _apply_adv_to_xfrc(self, adv_act):
        assert adv_act.shape[0] >= self.force_dim
        # get force mask
        new_xfrc = self.env.sim.data.xfrc_applied*0.0
        # apply forces at contact points
        for i, bindex in enumerate(self._adv_body_indicees):
            if self.force_dim == 1: 
                new_xfrc[bindex] = self.adv_fraction * np.array([adv_act[i], 0., 0., 0., 0., 0.])
            elif self.force_dim == 2: 
                new_xfrc[bindex] = self.adv_fraction * np.array([adv_act[i*2], 0., adv_act[i*2+1], 0., 0., 0.])
            elif self.force_dim == 3: 
                new_xfrc[bindex] = self.adv_fraction * np.array([adv_act[i*3], adv_act[i*3+1], adv_act[i*3+2], 0., 0., 0.])
            elif self.force_dim == 4: 
                new_xfrc[bindex] = self.adv_fraction * np.array([adv_act[i*4], adv_act[i*4+1], adv_act[i*4+2], adv_act[i*4+3], 0., 0.])
            elif self.force_dim == 5: 
                new_xfrc[bindex] = self.adv_fraction * np.array([adv_act[i*5], adv_act[i*5+1], adv_act[i*5+2], adv_act[i*5+3], 0., adv_act[i*5+4]])
            elif self.force_dim == 6: 
                new_xfrc[bindex] = self.adv_fraction * np.array([adv_act[i*6], adv_act[i*6+1], adv_act[i*6+2], adv_act[i*6+3], adv_act[i*6+4], adv_act[i*6+5]])
            else: 
                raise ValueError(f"Force dimension must be within [1, 6], not: ", self.force_dim)

        self.env.sim.data.xfrc_applied[:] = new_xfrc


    def sample_action(self):
        class pro_adv_action(object):
            def __init__(self, pro_action, adv_action):
                self.pro_action = pro_action
                self.adv_action = adv_action

        return pro_adv_action(self.action_space.sample(), self.adv_action_space.sample())


    def step(self, action):
        if hasattr(action, '__dict__'):
            print("action has dict")
            # apply force to system
            self._apply_adv_to_xfrc(action.adv_action)
            # perform step
            obs, rew, done, info = self.env.step(action.pro_action)
            self._last_obs = obs
            self._last_episode_starts = done
            return obs, rew, done, info
        else:
            if not self.operating_mode:
                # no adversarial influence
                obs, rew, done, info = self.env.step(action)
                self._last_obs = obs
                self._last_episode_starts = done
                return obs, rew, done, info

            elif self.operating_mode.lower() == "protagonist":
                # sample action from adversary
                with torch.no_grad():
                    # convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    if len(obs_tensor.shape) == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    action_sampled  = self._adv_policy._predict(obs_tensor, deterministic=True)

                clipped_actions = action_sampled.cpu().numpy()

                # clip the actions to avoid out of bound error
                if isinstance(self._adv_action_space, gym.spaces.Box):
                    clipped_actions = np.clip(clipped_actions, self._adv_action_space.low, self._adv_action_space.high).squeeze(0)
                
                # apply force to system
                self._apply_adv_to_xfrc(clipped_actions)
                # perform step
                obs, rew, done, info = self.env.step(action)

                self._last_obs = obs
                self._last_episode_starts = done

                return obs, rew, done, info
                
            elif self.operating_mode.lower() == "adversary":
                # sample action from protagonist
                with torch.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    if len(obs_tensor.shape) == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    action_sampled = self._pro_policy._predict(obs_tensor, deterministic=True)

                clipped_actions = action_sampled.cpu().numpy()

                # Clip the actions to avoid out of bound error
                if isinstance(self._action_space, gym.spaces.Box):
                    clipped_actions = np.clip(clipped_actions, self._action_space.low, self._action_space.high).squeeze(0)
                
                # apply force to system
                self._apply_adv_to_xfrc(action)
                # perform step
                obs, rew, done, info = self.env.step(clipped_actions)

                self._last_obs = obs
                self._last_episode_starts = done

                return obs, rew, done, info
                
            else:
                raise ValueError(f"Please choose operating mode either 'protagonist' or 'adversary', not: ", self.operating_mode)


    @property
    def observation_space(self):
        return self._observation_space


    @property
    def action_space(self):
        return self._action_space


    @property
    def adv_action_space(self):
        return self._adv_action_space
    

    def set_action_space(self, updated_action_space): 
        self._action_space = updated_action_space
    

    def set_adv_action_space(self, updated_adv_action_space): 
        self._adv_action_space = updated_adv_action_space


    def reset(self):
        self._last_obs = self.env.reset()
        return self._last_obs


class ObsNoiseWrapper(gym.ObservationWrapper):
    """
    Adding Gaussian noise to observations.
    """
    def __init__(self, env: Env, mu: float = 0, std: float = 1):
        super(ObsNoiseWrapper, self).__init__(env)
        self.env = env
        self.mu = mu
        self.std = std

    def observation(self, observation):
        """Apply noise to observation signal

        Args:
            observations (array): observation signal

        Returns:
            array: observation signal with Gaussian noise
        """
        for obs in observation:
            obs += np.random.normal(self.mu, self.std)
        return observation