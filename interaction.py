import atexit
import os
import random
import signal
import sys
import datetime
import threading
from typing import Callable, Union, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from yaml import warnings

from logger import DEFAULT_LOGGER_KEY, LogType, Logger
from s2v_wdn_dqn.agents.dueling_scorer_noisy_R2D2.dqn_agent import DQNAgent
from s2v_wdn_dqn.envs.wdn_env import WDNEnv
from matplotlib import pyplot as plt
from global_state import State

np.seterr(divide="raise", invalid="raise", over="raise", under="ignore")

#inp = "nodes_and_inps/L-TOWN_Real.inp"
#inp = "nodes_and_inps/STAR6.inp"
#inp = "nodes_and_inps/GRID.inp"
#inp = "nodes_and_inps/NET_2.inp"
#inp = "nodes_and_inps/5x5.inp"
#inp = "nodes_and_inps/10x10.inp"
#inp = "nodes_and_inps/20x20_branched.inp"
#inp = "nodes_and_inps/20x20.inp"
#inp = "nodes_and_inps/30x30.inp"
#inp = "nodes_and_inps/CSA_Base_reduced.inp"
#inp = "nodes_and_inps/random_generated_network.inp"
episodes = 1000
#eps_start = 0.75
#eps_end = 0.01
#eps_decay = 0.999
validate_each = -1
n_validate = 100
double_dqn = True
#global_step_counter = 0
#eps = eps_start
train_model = True
sensors_percentage = 0.90 #1.0 #0.95

base_seed = 666
random.seed(base_seed)
np.random.seed(base_seed)
torch.manual_seed(base_seed)
torch.cuda.manual_seed_all(base_seed)
#
#torch.use_deterministic_algorithms(True)
#os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required by torch 1.8+
#
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True

#torch.use_deterministic_algorithms(True)
#torch.autograd.set_detect_anomaly(True)


def get_epsilon(step, eps_start=1.0, eps_end=0.1, decay_steps=10000):
    # linear decay
    frac = min(1.0, step / decay_steps)
    return eps_start + frac * (eps_end - eps_start)

    
def run_episode(agent: DQNAgent,
                env: WDNEnv,
                episode_id: int,
                train_mode: bool = True) -> float:
    """
    Run one episode: agent interacts with env until done.
    Returns cumulative reward."""
    #global global_step_counter

    while True:    
        state, edge_feats, edge_status, global_feats = env.reset()
        agent.reset_episode()
        total_reward = 0.0
        done = False
        num_steps = 0
        failed = False

        while not done:
            action = agent.act(state, edge_feats, edge_status, global_feats, training=train_mode)
            info = env.step(action)

            if info is None:
                print("Environment step failed, ending episode.")
                #global_step_counter -= num_steps  # rollback step count
                failed = True
                break

            #if train_mode:
            #    global_step_counter += 1

            #print(f"After step: reward={info.reward}, done={info.done}")
            (next_state, next_edge_feats, next_edge_status, next_global_feats), reward, done = info.observation, info.reward, info.done
            if train_mode:
                agent.step(state, edge_feats, edge_status, global_feats, action,
                        reward, next_state, next_edge_feats, next_edge_status, next_global_feats, done, episode_id, num_steps)
            total_reward += reward
            num_steps += 1
            if num_steps % 50 == 0:
                print(f"  Step {num_steps} - Cumulative reward: {total_reward:.2f}")
            state, edge_feats, edge_status, global_feats = next_state, next_edge_feats, next_edge_status, next_global_feats

        if not failed:
            return total_reward, num_steps


def train(env: WDNEnv,
          agent: DQNAgent,
          n_episodes: int = 500,
          validate_each: Optional[int] = None,
          n_validate: int = 10,
          scheduler: Optional[Union[LambdaLR, Callable]] = None) -> list:
    """
    Train agent for a number of episodes, optionally validate periodically.
    Returns list of training returns."""
    #global global_step_counter
    rewards = []
    stepss = []
    for ep in range(1, n_episodes + 1):
        #eps = get_epsilon(global_step_counter)
        #print(f"Episode {ep}/{n_episodes} - Epsilon: {eps:.3f}")
        agent.train()
        reward, steps = run_episode(agent, env, ep, train_mode=True)
        agent.eval()
        rewards.append(reward)
        stepss.append(steps)

        # update LR if scheduler provided
        #if isinstance(scheduler, LambdaLR):
        #    scheduler.step()
        #elif callable(scheduler):
        #    lr = scheduler(ep)
        #    for g in agent.optimizer.param_groups:
        #        g['lr'] = lr
        # logging
        if ep % 500 == 0:
            print(f"[Episode {ep}/{n_episodes}] Reward: {reward:.2f}, Steps: {env.step_count:.1f}s")

        State.get(DEFAULT_LOGGER_KEY).log(f"[{datetime.datetime.now()}][Episode {ep}/{n_episodes}] Reward: {reward:.2f}, Steps: {env.step_count:.1f}s, Actions: {len(env.episode_actions)} Unique, {env.episode_actions}", msg_type=LogType.TRAINING)
        # optional periodic validation
        #if validate_each and ep % validate_each == 0:
        #    eval_returns = evaluate(env, agent, n_episodes=n_validate)

    return rewards, stepss


@torch.no_grad()
def evaluate(env: WDNEnv,
             agent: DQNAgent,
             n_episodes: int = 50) -> list:
    """
    Evaluate agent for a number of episodes.
    Returns list of evaluation returns."""
    agent.eval()
    returns = []
    stepss = []
    for ep in range(n_episodes):
        reward, steps = run_episode(agent, env, train_mode=False, episode_id=1000 + ep)
        returns.append(reward)
        stepss.append(steps)

    avg_return = np.mean(returns)
    avg_steps = np.mean(stepss)
    max_steps = np.max(stepss)
    min_steps = np.min(stepss)
    max_return = np.max(returns)
    min_return = np.min(returns)

    #plot distribution of returns using matplotlib
    plt.figure(figsize=(10,6))
    plt.hist(returns, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Episode Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.axvline(avg_return, color='red', linestyle='dashed', linewidth=1)
    plt.text(avg_return*1.05, plt.ylim()[1]*0.9, f'Avg: {avg_return:.2f}', color='red')
    plt.show()

    #plot distribution of steps using matplotlib
    plt.figure(figsize=(10,6))
    plt.hist(stepss, bins=50, color='green', alpha=0.7)
    plt.title('Distribution of Episode Steps')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.axvline(avg_steps, color='red', linestyle='dashed', linewidth=1)
    plt.text(avg_steps*1.05, plt.ylim()[1]*0.9, f'Avg: {avg_steps:.1f}', color='red')
    plt.show()

    print(f"Evaluation over {n_episodes} episodes: avg return {avg_return:.2f}, min {min_return:.2f}, max {max_return:.2f}; avg steps {avg_steps:.1f} (min {min_steps}, max {max_steps})")
    agent.train()
    return returns, stepss


def random_test(env: WDNEnv):
    rewards = []
    E = len(env.edge_list)
    for ep in range(100):
        state, edge_feats, edge_status, global_feats = env.reset()
        done = False
        total = 0
        while not done:
            action = np.random.randint(E)
            state, reward, done = env.step(action)
            total += reward
        rewards.append(total)
    print("Random policy episode returns:", rewards)
    print("Random policy avg return:", np.mean(rewards), "std:", np.std(rewards))




if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Train DQN on WDNEnv")
    #parser.add_argument("--inp", type=str, default="nodes_and_inps/L-TOWN_Real.inp",
    #                    help="EPANET .inp file path")
    #parser.add_argument("--episodes", type=int, default=10000,
    #                    help="Training episodes")
    #parser.add_argument("--eps_start", type=float, default=1.0)
    #parser.add_argument("--eps_end", type=float, default=0.01)
    #parser.add_argument("--eps_decay", type=float, default=0.999,
    #                    help="If <1: multiplicative decay; if >=1: subtractive step")
    #parser.add_argument("--n_validate", type=int, default=0,
    #                    help="Number of validation episodes per eval; 0 to disable")
    #parser.add_argument("--double_dqn", type=bool, default=True,
    #                    help="Enable Double DQN")
    #args = parser.parse_args()

    # Set up environment and agent

    sensors_coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    demands_percentages = [0.05]
    num_leaks = [1, 2, 3]
    inps = ["nodes_and_inps/20x20_branched.inp"]
    
    _exit_handler_lock = threading.Lock()
    _exit_handler_called = False

    def _exit_handler(signum=None, frame=None, close_program=True):
        global _exit_handler_called
        with _exit_handler_lock:
            if _exit_handler_called:
                return
            if close_program:
                _exit_handler_called = True


        log_name = State.get("current_log_name", "unknown_run")
        agent = State.get("current_env").agent
        _save_on_exit(log_name, agent)
        if signum is not None:
            sys.exit(0)

    def _save_on_exit(log_name="unknown_run", agent=None):
        if agent is None:
            return
        agent.save_model('checkpoints/checkpoint.pth')
        agent.save_model(f'checkpoints/{log_name}_checkpoint.pth')
        
    atexit.register(_exit_handler)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _exit_handler)

    for inp in inps:
        for sensors_percentage in sensors_coverages:
            for n_leaks in num_leaks:
                for demand_percentage in demands_percentages:
                    #if sensors_percentage == 1.0 and n_leaks == 1:
                    #    print("skippo primo caso")
                    #    continue                    
                    log_name = f"Training_WDN_DQN_{os.path.basename(inp).replace('.inp','')}_sensors{int(sensors_percentage*100)}_demands{int(demand_percentage*100)}_leaks{n_leaks}"
                    State.set(DEFAULT_LOGGER_KEY, Logger.get(name=log_name, env=True, agent=True, episode=True, loss=True, td_errors=True, q_values=True, grad_norm=True, training=True))
                    #State.get(DEFAULT_LOGGER_KEY).log(f"Starting run with inp={inp}, sensors_percentage={sensors_percentage}, demand_percentage={demand_percentage}", msg_type=LogType.TRAINING)
                    State.set("current_log_name", log_name)

                    print("### Starting run with inp={}, sensors_percentage={}, demand_percentage={}, n_leaks={}".format(inp, sensors_percentage, demand_percentage, n_leaks))

                    env = WDNEnv(simulation_file_path=inp,
                                normalize_reward=True,
                                double_dqn=True,
                                global_timestep=60,
                                simulation_duration=4 * 3600,
                                sensors_percentage=sensors_percentage,
                                demand_percentage=demand_percentage,
                                num_leaks=n_leaks)  # 6 hours in seconds
                    agent = env.agent

                    State.set("current_env", env)

                    #env.simulation.plot_network()
                    #from sys import exit
                    #exit(0)


                    if os.path.exists('checkpoints/checkpoint.pth'):
                        print("Loading existing model from checkpoints/checkpoint.pth")
                        agent.load_model('checkpoints/checkpoint.pth')
                    else:
                        print("No checkpoints/checkpoint.pth found, starting fresh training")


                    

                    # Optionally set up LR scheduler
                    # example: scheduler = LambdaLR(agent.optimizer, lr_lambda)
                    scheduler = None

                    rewards, steps = train(env, agent, n_episodes=episodes, validate_each=(validate_each if validate_each > 0 else None), n_validate=n_validate, scheduler=scheduler)
                    
                    log_name = f"Evaluation_WDN_DQN_{os.path.basename(inp).replace('.inp','')}_sensors{int(sensors_percentage*100)}_demands{int(demand_percentage*100)}_leaks{n_leaks}"
                    State.set(DEFAULT_LOGGER_KEY, Logger.get(name=log_name, env=True, agent=True, episode=True, loss=True, td_errors=True, q_values=True, grad_norm=True, training=True))
                    rewards, steps = evaluate(env, agent, n_episodes=200)
                        
                    _exit_handler(close_program=False)
