import argparse
import random
import json
import os
from pathlib import Path

import numpy as np
import torch

from world.delivery_environment import DeliveryEnvironment
from agents.DQN_agent import DQNAgent

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_dqn(env, agent, episodes=10, max_steps=None, record_path=False):
    """
    Run evaluation with epsilon=0.
    Returns:
      - success_rate, avg_return, avg_steps
      - If record_path=True: avg_path_len, avg_efficiency, sample_traj
    """
    # 1) Freeze epsilon
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    # 2) Prepare lists
    succs, rets, steps = [], [], []
    path_lens, effs = [], []
    sample_traj = None

    for _ in range(episodes):
        state = env.reset()
        start  = env.agent_pos.copy()
        target = env.tables[env.target_table_idx]
        traj   = [start.copy()] if record_path else None

        total_r, done = 0.0, False
        for t in range(max_steps or env.max_steps):
            # Correct action selection
            a = agent.select_action(state)
            next_state, r, done, info = env.step(a)
            total_r += r
            if record_path:
                traj.append(env.agent_pos.copy())
            state = next_state
            if done:
                break

        # 3) Accumulate stats
        succs.append(int(info.get("success", False)))
        steps.append(t + 1)
        rets.append(total_r)

        # 4) Store first trajectory for visualization
        if record_path and sample_traj is None:
            sample_traj = (traj, start, target)

        # 5) Compute path length & efficiency for this episode
        if record_path:
            disps = np.diff(np.array(traj), axis=0)
            path_len = np.linalg.norm(disps, axis=1).sum()
            straight = np.linalg.norm(start - target)
            efficiency = path_len / straight if straight > 0 else np.nan
            path_lens.append(path_len)
            effs.append(efficiency)

    # 6) Restore epsilon
    agent.epsilon = old_eps

    # 7) Prepare result dict
    result = {
        "success_rate": np.mean(succs),
        "avg_return":   np.mean(rets),
        "avg_steps":    np.mean(steps),
    }
    if record_path:
        result.update({
            "avg_path_len":  np.mean(path_lens),
            "avg_efficiency":np.nanmean(effs),
            "sample_traj":   sample_traj
        })

    return result

def main():
    parser = argparse.ArgumentParser("Train DQN on DeliveryEnvironment")
    parser.add_argument("--env-config", type=Path, default=None,
                        help="npz file for layout")
    parser.add_argument("--episodes", type=int, default=500,
                        help="number of training episodes")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="max steps per episode")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.01)
    parser.add_argument("--eps-decay", type=float, default=0.995)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--target-update", type=int, default=1000,
                        help="learn steps between target net sync")
    parser.add_argument("--log-interval", type=int, default=20,
                        help="episodes between logging")
    parser.add_argument("--eval-interval", type=int, default=50,
                        help="episodes between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="episodes per evaluation")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="episodes between model saves")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="directory for logs, checkpoints, figures")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")
    parser.add_argument("--no-gui", action="store_true",
                        help="disable environment GUI")
    args = parser.parse_args()

    # make output dirs
    ckpt_dir = args.output_dir / "checkpoints"
    log_dir  = args.output_dir / "logs"
    fig_dir  = args.output_dir / "figures"
    for d in (ckpt_dir, log_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    # fix seeds
    set_global_seed(args.seed)

    # init environment and agent
    env = DeliveryEnvironment(space_file=args.env_config,
                              enable_gui=not args.no_gui,
                              max_steps=args.max_steps)
    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        epsilon_decay=args.eps_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update
    )


    print(f"Using device: {agent.device}")
    # training loop
    logs = []
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        ep_return, ep_steps, ep_success = 0.0, 0, False

        for t in range(args.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            ep_return += reward
            ep_steps += 1
            if info.get("success", False):
                ep_success = True

            agent.observe_transition(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            
        agent.decay_epsilon()

        logs.append({
            "episode": ep,
            "return": ep_return,
            "steps": ep_steps,
            "success": int(ep_success),
            "epsilon": agent.epsilon
        })

        # logging
        if ep % args.log_interval == 0:
            recent = logs[-args.log_interval:]
            avg_ret  = np.mean([x["return"] for x in recent])
            avg_succ = np.mean([x["success"] for x in recent])
            avg_stp  = np.mean([x["steps"] for x in recent])
            print(f"[Ep {ep}] return={avg_ret:.1f} succ={avg_succ:.2%} "
                  f"steps={avg_stp:.1f} eps={agent.epsilon:.3f}")

        # evaluation
        if ep % args.eval_interval == 0:
            res = evaluate_dqn(env, agent,
                               episodes=args.eval_episodes,
                               max_steps=args.max_steps,
                               record_path=True)
            print(f"@Ep{ep}: succ={res['success_rate']:.2%} "
                  f"ret={res['avg_return']:.1f} steps={res['avg_steps']:.1f} "
                  f"eff={res['avg_efficiency']:.2f}")

        # save model
        if ep % args.save_interval == 0:
            path = ckpt_dir / f"dqn_ep{ep}.pth"
            agent.save(path)

    # final save
    agent.save(ckpt_dir / "dqn_final.pth")
    with open(log_dir / "train_log.json", "w") as f:
        json.dump(logs, f, indent=2)

    print("Training complete.")
    env.close()

if __name__ == "__main__":
    main()
