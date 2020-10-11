import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.env.wrappers.recsim_wrapper import make_recsim_env

from slateq_from_scratch.userchoice import UserChoiceModel
from slateq_from_scratch.qmodel import QModel


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

    q_model = QModel()
    q_optimizer = torch.optim.Adam(q_model.parameters())
    q_loss_fn = nn.MSELoss(reduction="sum")

    for idx_episode in range(1000):
        episode_reward = 0.0
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
        episode_step = 0
        while not done:
            entry = {}
            entry["state_user"] = pack_state_user(obs)
            entry["state_doc"] = pack_state_doc(obs)

            action = compute_action(obs, user_choice_model, q_model)

            entry["action"] = np.array(action, dtype=np.int32)

            episode_step += 1
            obs, reward, done, info = env.step(action)
            if last_entry:
                last_entry["next_action"] = np.array(action, dtype=np.int32)
                last_entry["next_state_user"] = pack_state_user(obs)
                last_entry["next_state_doc"] = pack_state_doc(obs)
                buf.add(last_entry)

            click_idx, myopic_reward = parse_response(obs["response"])
            episode_reward += myopic_reward

            # if not done:
            #     assert click_idx >= 0, (step, obs, reward, done, info)

            entry["click"] = np.array(click_idx, dtype=np.float32)
            entry["myopic_reward"] = np.array(myopic_reward, dtype=np.float32)

            last_entry = entry

        print(episode_reward, episode_step)

        if (idx_episode + 1) % 10 == 0:
            train_user_choice_model(
                user_choice_model, loss_fn, optimizer, buf, batch_size=4, num_iters=100
            )
            train_q_model(
                q_model,
                user_choice_model,
                q_loss_fn,
                q_optimizer,
                buf,
                batch_size=4,
                num_iters=30,
            )


def train_q_model(
    q_model, user_choice_model, q_loss_fn, q_optimizer, buf, batch_size, num_iters
):
    tot_loss = 0
    tot_items = 0
    for _ in range(num_iters):
        batch = buf.sample(batch_size)

        next_selected_doc = torch.cat(
            [
                torch.index_select(doc, 0, sel).unsqueeze(0)
                for doc, sel in zip(
                    batch["next_state_doc"], batch["next_action"].long()
                )
            ],
            dim=0,
        )  # shape=[batch_size, slate_size, num_embeddings]
        next_user = batch["next_state_user"]
        with torch.no_grad():
            q_values = q_model(
                next_user, next_selected_doc
            )  # shape=[batch_size, slate_size]
            scores = user_choice_model(
                next_user, next_selected_doc
            )  # shape=[batch_size, slate_size+1]
            scores = torch.exp(scores - torch.max(scores, dim=1, keepdim=True)[0])
            next_q_values = torch.sum(q_values * scores[:, :-1], dim=1) / torch.sum(
                scores, dim=1
            )  # shape=[batch_size]
        target_q_values = next_q_values + batch["myopic_reward"]  # shape=[batch_size]

        selected_doc = torch.cat(
            [
                torch.index_select(doc, 0, sel).unsqueeze(0)
                for doc, sel in zip(batch["state_doc"], batch["action"].long())
            ],
            dim=0,
        )  # shape=[batch_size, slate_size, num_embeddings]
        user = batch["state_user"]
        q_values = q_model(user, selected_doc)  # shape=[batch_size, slate_size]
        scores = user_choice_model(
            user, selected_doc
        )  # shape=[batch_size, slate_size+1]
        scores = torch.exp(scores - torch.max(scores, dim=1, keepdim=True)[0])
        q_values = torch.sum(q_values * scores[:, :-1], dim=1) / torch.sum(
            scores, dim=1
        )  # shape=[batch_size]

        loss = q_loss_fn(q_values, target_q_values)
        q_optimizer.zero_grad()
        loss.backward()
        q_optimizer.step()
        tot_loss += loss.item()
        tot_items += batch_size
    print(tot_loss / tot_items)


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


def compute_action_random(obs, user_choice_model):
    n_docs = len(obs["doc"])
    selected = np.random.choice(list(range(n_docs)), 3)
    action = tuple(selected)
    return action


def compute_action_choice_model(obs, user_choice_model, q_model):
    """Select docs with highest click probability"""
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


def compute_action(
    obs, user_choice_model: UserChoiceModel, q_model: QModel, slate_size: int = 3
):
    user = pack_state_user(obs)
    doc = pack_state_doc(obs)
    user = torch.tensor(user).unsqueeze(0)
    doc = torch.tensor(doc).unsqueeze(0)

    with torch.no_grad():
        scores = user_choice_model(user, doc).squeeze(0)  # shape=[num_docs+1]
        scores = torch.exp(scores - torch.max(scores, dim=-1)[0])
        scores_doc = scores[:-1]  # shape=[num_docs]
        score_no_click = scores[-1]  # shape=[]
        q_values = q_model(user, doc).squeeze(0)  # shape=[num_docs]

    num_docs = len(obs["doc"])
    indices = torch.tensor(np.arange(num_docs)).long()
    slates = torch.combinations(indices, r=slate_size)  # shape=[num_slates, num_docs]
    num_slates, _ = slates.shape

    slate_decomp_q_values = torch.gather(
        q_values.unsqueeze(0).expand(num_slates, num_docs), 1, slates
    )  # shape=[num_slates, slate_size]
    slate_scores = torch.gather(
        scores_doc.unsqueeze(0).expand(num_slates, num_docs), 1, slates
    )  # shape=[num_slates, slate_size]
    slate_q_values = (slate_decomp_q_values * slate_scores).sum(dim=1) / (
        slate_scores.sum(dim=1) + score_no_click
    )  # shape=[num_slates]

    idx = np.argmax(slate_q_values.detach().numpy())
    selected = slates[idx].detach().numpy().tolist()
    action = tuple(selected)
    return action


def parse_response(response):
    for idx, row in enumerate(response):
        if row["click"] == 1:
            return idx, row["watch_time"]
    return len(response), 0.0


if __name__ == "__main__":
    main()
