from typing import Iterable, Optional, Union

import gym
import gym.spaces
import numpy as np

from wsrl.data.dataset import Dataset, DatasetDict, _sample
from wsrl.envs.env_common import calc_return_to_go


def _init_replay_dict(obs_space: gym.Space, capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert (
            dataset_dict.keys() == data_dict.keys()
        ), f"{dataset_dict.keys()} != {data_dict.keys()}"
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=bool),
            dones=np.empty((capacity,), dtype=np.float32),
        )

        super().__init__(dataset_dict, seed)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self._sequential_index = 0
        self.unsampled_indices = list(range(self._size))
        self._discount = discount

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)


class ReplayBufferMC(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
        env_name: str = None,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
    ):
        assert discount is not None, "ReplayBufferMC requires a discount factor"
        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space,
            seed,
            discount,
        )

        mc_returns = np.empty((capacity,), dtype=np.float32)
        self.dataset_dict["mc_returns"] = mc_returns

        if env_name is None:
            raise ValueError("ReplayBufferMC requires env_name")
        self.env_name = env_name
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self._allow_idxs = []
        self._traj_start_idx = 0

    def insert(self, data_dict: DatasetDict):
        # assumes replay buffer capacity is more than the number of online steps
        assert self._size < self._capacity, "replay buffer has reached capacity"

        data_dict["mc_returns"] = None
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        # if "dones" not in data_dict:
        #     data_dict["dones"] = 1 - data_dict["masks"]

        if data_dict["dones"] == 1.0:
            # compute the mc_returns
            rewards = self.dataset_dict["rewards"][self._traj_start_idx : self._insert_index + 1]
            masks = self.dataset_dict["masks"][self._traj_start_idx : self._insert_index + 1]
            self.dataset_dict["mc_returns"][self._traj_start_idx : self._insert_index + 1] = (
                calc_return_to_go(
                    self.env_name,
                    rewards,
                    masks,
                    self._discount,
                    reward_scale=self.reward_scale,
                    reward_bias=self.reward_bias,
                )
            )

            self._allow_idxs.extend(list(range(self._traj_start_idx, self._insert_index + 1)))
            self._traj_start_idx = self._insert_index + 1

        self._size += 1
        self._insert_index += 1

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> dict:
        if indx is None:
            indx = self.np_random.choice(self._allow_idxs, size=batch_size, replace=True)
        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            batch[k] = _sample(self.dataset_dict[k], indx)

        return batch
