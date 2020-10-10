import random
from collections import defaultdict

import numpy as np
from ray.rllib.env.wrappers.recsim_wrapper import make_recsim_env


class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.keys = (
            "state_user,state_doc,action,click,myopic_reward,"
            "next_state_user,next_state_doc,next_action"
        ).split(",")

    def add(self, entry):
        self.buffer.append(tuple(entry[key] for key in self.keys))

    def sample(self, batch_size: int = 64):
        sampled = random.sample(self.buffer, batch_size)
        return {
            key: np.array([row[idx] for row in sampled])
            for idx, key in enumerate(self.keys)
        }


class UserChoiceModel(nn.Module):
    pass


def main():
    buf = ReplayBuffer()
    env = make_recsim_env({"slate_size": 3})
    for _ in range(30):
        obs = env.reset()
        done = False
        entry = {
            "state_user": None,
            "state_doc": None,
            "action": None,
            "click": None,
            "myopic_reward": None,
            "next_state_user": None,
            "next_state_doc": None,
            "next_action": None,
        }
        last_entry = {}
        step = 0
        while not done:
            entry = {}
            entry["state_user"] = pack_state_user(obs)
            entry["state_doc"] = pack_state_doc(obs)

            action = compute_action(obs)

            entry["action"] = np.array(action, dtype=np.int32)

            step += 1
            obs, reward, done, info = env.step(action)
            if last_entry:
                last_entry["next_action"] = np.array(action, dtype=np.int32)
                last_entry["next_state_user"] = pack_state_user(obs)
                last_entry["next_state_doc"] = pack_state_doc(obs)
                buf.add(last_entry)

            click_idx, myopic_reward = parse_response(obs["response"])

            # if not done:
            #     assert click_idx >= 0, (step, obs, reward, done, info)

            entry["click"] = np.array(click_idx, dtype=np.float32)
            entry["myopic_reward"] = np.array(myopic_reward, dtype=np.float32)

            last_entry = entry

    batch = buf.sample(5)
    for k, v in batch.items():
        print(k, v.shape, v.dtype)


def pack_state_user(obs):
    return obs["user"].astype(np.float32)


def pack_state_doc(obs):
    return np.array(list(obs["doc"].values()), dtype=np.float32)


def compute_action(obs):
    user = obs["user"]
    doc = np.stack(list(obs["doc"].values()), axis=0)
    scores = np.einsum("c,dc->d", user, doc)
    n_docs = len(scores)
    selected = np.random.choice(list(range(n_docs)), 3)
    action = tuple(selected)
    return action


def parse_response(response):
    for idx, row in enumerate(response):
        if row["click"] == 1:
            return idx, row["watch_time"]
    return len(response), 0.0


if __name__ == "__main__":
    main()
