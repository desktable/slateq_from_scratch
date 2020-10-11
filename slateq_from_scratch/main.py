import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.env.wrappers.recsim_wrapper import make_recsim_env
from slateq_from_scratch.userchoice import UserChoiceModel


class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.keys = (
            "state_user,state_doc,action,click,myopic_reward,"
            "next_state_user,next_state_doc,next_action"
        ).split(",")

    def add(self, entry):
        self.buffer.append(tuple(entry[key] for key in self.keys))

    def sample(self, batch_size: int = 64, to_tensor: bool = True):
        sampled = random.sample(self.buffer, batch_size)
        ret = {
            key: np.array([row[idx] for row in sampled])
            for idx, key in enumerate(self.keys)
        }
        if to_tensor:
            ret = {key: torch.tensor(val) for key, val in ret.items()}
        return ret


def main():
    buf = ReplayBuffer()
    env = make_recsim_env({"slate_size": 3})

    user_choice_model = UserChoiceModel()
    optimizer = torch.optim.Adam(user_choice_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    for idx_episode in range(1000):
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

            action = compute_action(obs, user_choice_model)

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
        if (idx_episode + 1) % 10 == 0:
            train_user_choice_model(
                user_choice_model, loss_fn, optimizer, buf, batch_size=16, num_iters=10
            )


def train_user_choice_model(model, loss_fn, optimizer, buf, batch_size, num_iters):
    tot_loss = 0
    tot_items = 0
    for _ in range(num_iters):
        batch = buf.sample(batch_size)
        selected_doc = torch.cat(
            [
                torch.index_select(doc, 0, sel).unsqueeze(0)
                for doc, sel in zip(batch["state_doc"], batch["action"].long())
            ],
            dim=0,
        )
        scores = model(batch["state_user"], selected_doc)
        loss = loss_fn(scores, batch["click"].long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        tot_items += batch_size
    print(tot_loss / tot_items, model.a.item(), model.b.item())


def pack_state_user(obs):
    return obs["user"].astype(np.float32)


def pack_state_doc(obs):
    return np.array(list(obs["doc"].values()), dtype=np.float32)


def compute_action_greedy(obs, user_choice_model):
    user = obs["user"]
    doc = np.stack(list(obs["doc"].values()), axis=0)
    scores = np.einsum("c,dc->d", user, doc)
    n_docs = len(scores)
    selected = np.random.choice(list(range(n_docs)), 3)
    action = tuple(selected)
    return action


def compute_action(obs, user_choice_model):
    user = pack_state_user(obs)
    doc = pack_state_doc(obs)
    user = torch.tensor(user).unsqueeze(0)
    doc = torch.tensor(doc).unsqueeze(0)
    with torch.no_grad():
        scores = user_choice_model(user, doc).squeeze(0)
        # scores = nn.Softmax(dim=-1)(scores)
        scores = scores.detach().numpy()
    scores_doc, score_no_click = scores[:-2], scores[-1]
    selected = np.argsort(scores_doc)[-3:].tolist()
    # print(scores_doc, selected, score_no_click)
    action = tuple(selected)
    return action


def parse_response(response):
    for idx, row in enumerate(response):
        if row["click"] == 1:
            return idx, row["watch_time"]
    return len(response), 0.0


if __name__ == "__main__":
    main()
