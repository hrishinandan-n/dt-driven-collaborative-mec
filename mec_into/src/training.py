"""Training Pipelines for INTO and Baseline DDQN Agents.

Executes Algorithm 2 across time slots, collecting transitions,
updating actor and critic networks, and tracking convergence metrics.
"""

from typing import List, Dict
import numpy as np
import yaml
from .environment import MECEnvironment
from .into_agent import INTOAgent
from .baselines.ddqn import DDQNBaseline


def train_into(
    config: dict,
    total_slots: int = 3000,
    eval_interval: int = 100,
    seed: int = 42,
    lr_override: float = None,
    gamma_override: float = None,
    checkpoint_path: str = None,
) -> Dict[str, list]:
    """Trains INTO agent following Algorithm 2 for a specified number of time slots."""
    if lr_override is not None:
        config["drl"]["learning_rate_actor"] = lr_override
        config["drl"]["learning_rate_critic"] = lr_override
    if gamma_override is not None:
        config["drl"]["discount_factor"] = gamma_override

    env = MECEnvironment(config, seed=seed)
    state = env.reset(seed=seed)
    state_dim = len(state)

    agent = INTOAgent(
        state_dim=state_dim,
        num_tasks=env.total_devices,
        num_bs=env.num_bs,
        config=config,
    )

    history = {
        "slots": [],
        "income": [],
        "average_income": [],
        "critic_loss": [],
        "actor_loss": [],
    }

    cumulative_income = 0.0

    for slot in range(1, total_slots + 1):
        # Line 5: Generate proto-action
        proto_action = agent.select_action(state, add_noise=True)

        # Lines 6-7: Refine action, execute on environment (Digital Twin feedback), observe r(t), s(t+1)
        next_state, income, _, info = env.step(proto_action)

        # Line 8: Store experience tuple
        agent.memory.push(state, proto_action.flatten(), income, next_state)

        # Lines 9-13: Updates
        c_loss, a_loss = agent.update()

        cumulative_income += income
        avg_income = cumulative_income / slot

        history["slots"].append(slot)
        history["income"].append(income)
        history["average_income"].append(avg_income)
        history["critic_loss"].append(c_loss)
        history["actor_loss"].append(a_loss)

        state = next_state

        if slot % eval_interval == 0 or slot == total_slots:
            print(f"[INTO] Slot {slot:4d}/{total_slots} | Avg Income: {avg_income:.2f} | Income: {income:.2f} | C-Loss: {c_loss:.4f}")

    if checkpoint_path:
        agent.save_checkpoint(checkpoint_path)

    return {"agent": agent, "history": history}


def train_ddqn(
    config: dict,
    total_slots: int = 3000,
    eval_interval: int = 100,
    seed: int = 42,
    checkpoint_path: str = None,
) -> Dict[str, list]:
    """Trains DDQN baseline agent for comparison."""
    env = MECEnvironment(config, seed=seed)
    state = env.reset(seed=seed)
    state_dim = len(state)

    agent = DDQNBaseline(
        state_dim=state_dim,
        num_tasks=env.total_devices,
        num_bs=env.num_bs,
        config=config,
    )

    history = {
        "slots": [],
        "income": [],
        "average_income": [],
        "loss": [],
    }

    cumulative_income = 0.0

    for slot in range(1, total_slots + 1):
        proto_action = agent.select_action(state, evaluate=False)
        next_state, income, _, info = env.step(proto_action)

        agent.memory.push(state, proto_action.flatten(), income, next_state)
        loss = agent.update()

        cumulative_income += income
        avg_income = cumulative_income / slot

        history["slots"].append(slot)
        history["income"].append(income)
        history["average_income"].append(avg_income)
        history["loss"].append(loss)

        state = next_state

        if slot % eval_interval == 0 or slot == total_slots:
            print(f"[DDQN] Slot {slot:4d}/{total_slots} | Avg Income: {avg_income:.2f} | Income: {income:.2f} | Loss: {loss:.4f}")

    if checkpoint_path:
        agent.save_checkpoint(checkpoint_path)

    return {"agent": agent, "history": history}
