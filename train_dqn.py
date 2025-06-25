from argparse import ArgumentParser
from pathlib import Path
import random, datetime, csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

# ---------------------------------------------------------------------- #
#  Project imports
# ---------------------------------------------------------------------- #
try:
    from world.enviroment_final import ContinuousEnvironment, COLLISION_PEN
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from world.enviroment_final import ContinuousEnvironment

from agents.dqn_final import DQNAgent

# ---------------------------------------------------------------------- #
def parse_args() -> ArgumentParser:
    p = ArgumentParser("DQN on Restaurant Environment")
    p.add_argument("--restaurant", type=Path,
                   default=Path("grid_configs/my_first_restaurant.npz"))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--load_model", type=Path)
    p.add_argument("--demo", action="store_true")
    return p.parse_args()

# ---------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    #  Reproducibility
    # ------------------------------------------------------------------ #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------ #
    #  Environment
    # ------------------------------------------------------------------ #
    env = ContinuousEnvironment(
        space_file=args.restaurant,
        enable_gui=not args.no_gui,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    #  Agent
    # ------------------------------------------------------------------ #
    agent = DQNAgent(
        state_size      = env.state_size,
        action_size     = env.action_size,
        map_width  = env.width,
        map_height = env.height,
        num_tables      = len(env.tables),        
        target_update_freq = 1000,
        gamma           = 0.99,
        batch_size      = 128,
        epsilon_start   = 1.0,
        epsilon_end     = 0.05,
        epsilon_decay   = 0.995,
        buffer_size     = 100_000,
        learning_rate   = 1e-4,
        device          = device,
        seed            = args.seed,
    )

    if args.load_model:
        agent.load(args.load_model)
    if args.demo:
        agent.epsilon = 0.0

    # ------------------------------------------------------------------ #
    #  CSV logger
    # ------------------------------------------------------------------ #
    run_id  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")     
    log_path = Path(f"training_log_{run_id}.csv")
    csv_f = log_path.open("a", newline="")
    csv_w = csv.writer(csv_f)
    if csv_f.tell() == 0:     
        csv_w.writerow([
            "episode", "timestamp",
            "reward", "length", "success",
            "epsilon", "loss", "buffer_size",
            "avg_q", "collisions"
        ])

    # ------------------------------------------------------------------ #
    #  Training hyper-parameters
    # ------------------------------------------------------------------ #
    MAX_STEPS_LONG, MAX_STEPS_SHORT = 4000, 1000
    LOG_EVERY = 20                               

    rewards, lengths, succ_flags = [], [], []
    obs = env.reset()

    with trange(args.episodes, desc="Episodes") as pbar:
        for ep in pbar:
            max_steps = MAX_STEPS_LONG if ep < 40 else MAX_STEPS_SHORT
            ep_r, ep_len, done = 0.0, 0, False
            deliveries_before = env.deliveries_done
            collisions = 0

            while not done:
                action = agent.action(obs)
                nxt, r, done, info = env.step(action)
                if info.get("collision"):
                    collisions += 1
                agent.observe(obs, action, r, nxt, done)

                obs, ep_r, ep_len = nxt, ep_r + r, ep_len + 1
                if ep_len >= max_steps:
                    done = True
                    
            if agent.epsilon > agent.epsilon_end:
                agent.epsilon *= agent.epsilon_decay

            # ---------- episode end stats ----------
            rewards.append(ep_r)
            lengths.append(ep_len)
            success = int(env.deliveries_done > deliveries_before)
            succ_flags.append(success)
            loss_val = getattr(agent, "last_loss", None)
            with torch.no_grad():
                    state_vec = torch.from_numpy(agent.state_to_vector(obs)) \
                                     .unsqueeze(0) \
                                     .to(agent.device)           
                    avg_q = float(agent.q_network(state_vec).mean().item()) 

            # write CSV
            csv_w.writerow([
                ep + 1,
                datetime.datetime.now().isoformat(timespec="seconds"),
                f"{ep_r:.2f}", ep_len, success,
                f"{agent.epsilon:.4f}",
                f"{loss_val:.4f}" if loss_val is not None else "",
                len(agent.replay_buffer),
                f"{avg_q:.3f}",                     
                collisions                          
            ])
            csv_f.flush()

            # -------- static log every LOG_EVERY episodes ----------
            if (ep + 1) % LOG_EVERY == 0 or ep == 0:
                window = slice(-LOG_EVERY, None)
                succ_cnt = int(sum(succ_flags[window]))
                avg_r    = np.mean(rewards[window])
                avg_len  = np.mean(lengths[window])

                tqdm.write(
                    f"[{ep+1:>4d}/{args.episodes}] "
                    f"avg_R={avg_r:7.1f} | avg_len={avg_len:4.0f} "
                    f"succ={succ_cnt}/{len(succ_flags[window])} "
                    f"eps={agent.epsilon:.3f}"
                )

            obs = env.reset()

    # ------------------------------------------------------------------ #
    #  Save model & curves
    # ------------------------------------------------------------------ #
    out_dir = Path("trained_model")
    agent.save(out_dir)
    print(f"Model saved to {out_dir/'dqn_model.pth'}")
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(rewards); plt.title("Episode Reward")
    plt.subplot(1, 2, 2); plt.plot(lengths);  plt.title("Episode Length")
    plt.tight_layout()
    plt.savefig(f"training_progress_{run_id}.png")   # ← timestamped filename
    # plt.show()   # optional: only works in GUI-capable environments
    plt.close()
    
    csv_f.close()
    env.close()

# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    main()