#!/usr/bin/env python
from sre_constants import SUCCESS
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv, MPECatchingEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
import time
import imageio

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPECatchingEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPECatchingEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def _t2n(x):
    return x.detach().cpu().numpy()

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError


    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents
    num_goods = all_args.num_good_agents
    num_bads = all_args.num_adversaries

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "num_goods": num_goods,
        "num_bads": num_bads,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    from onpolicy.runner.separated.catching_runner import MPERunner as Runner

    runner = Runner(config)
    # runner.render()
    num_test_episode = 10
    episode_length = all_args.episode_length

    # env
    single_env = MPECatchingEnv(all_args)

    # start 
    for ep in range(num_test_episode):
        # two group reward
        success_step = 0
        eval_episode_rewards_good = []
        eval_episode_rewards_bad = []
        all_frames = []

        if all_args.save_gifs:
                image = single_env.render()
                all_frames.append(image)
         
        # reset
        eval_obs = single_env.reset()

        # rnn hidden state
        all_eval_rnn_states = [np.zeros((1, *runner.buffer[0].rnn_states.shape[2:]), dtype=np.float32),
                               np.zeros((1, *runner.buffer[1].rnn_states.shape[2:]), dtype=np.float32)]
        all_eval_masks = [np.ones((1, runner.num_bads, 1), dtype=np.float32),
                            np.ones((1, runner.num_goods, 1), dtype=np.float32)]
        
        for step in range(episode_length):
            all_eval_actions_env = []
            calc_start = time.time()
            for group_id in range(runner.num_groups):
                runner.trainer[group_id].prep_rollout()
                group_obs = np.array(eval_obs[:runner.num_bads]) if group_id == 0 else np.array(eval_obs[runner.num_bads:])

                eval_action, eval_rnn_states = runner.trainer[group_id].policy.act(group_obs,
                                                np.concatenate(all_eval_rnn_states[group_id]),
                                                np.concatenate(all_eval_masks[group_id]),
                                                deterministic=True)
                
                eval_actions = np.array([_t2n(eval_action)])
                eval_rnn_states = _t2n(eval_rnn_states)
                all_eval_rnn_states[group_id][0] = np.copy(eval_rnn_states)

                if runner.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(runner.envs.action_space[0].shape):
                        eval_uc_actions_env = np.eye(runner.envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                        if i == 0:
                            eval_actions_env = eval_uc_actions_env
                        else:
                            eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
                elif runner.envs.action_space[0].__class__.__name__ == 'Discrete':
                    eval_actions_env = np.squeeze(np.eye(runner.envs.action_space[0].n)[eval_actions], 2)
                else:
                    raise NotImplementedError
            
                all_eval_actions_env.append(eval_actions_env)
                    
            # gather  all groups' action into one array
            actions_env = np.concatenate(all_eval_actions_env, axis=1)[0].tolist()

            # evaluate
            eval_obs, eval_rewards, eval_dones, eval_infos = single_env.step(actions_env)

            for group_id in range(runner.num_groups):
                # split into two groups
                group_ev_rewards = np.array([eval_rewards[:runner.num_bads]]) if group_id == 0 else np.array([eval_rewards[runner.num_bads:]])
                group_ev_dones = np.array([eval_dones[:runner.num_bads]]) if group_id == 0 else np.array([eval_dones[runner.num_bads:]])

                all_eval_rnn_states[group_id][group_ev_dones == True] = np.zeros(((group_ev_dones == True).sum(), runner.recurrent_N, runner.hidden_size), dtype=np.float32)
                
                num_inner_agent = runner.num_bads if group_id == 0 else runner.num_goods
                masks = np.ones((1, num_inner_agent, 1), dtype=np.float32)
                masks[group_ev_dones == True] = np.zeros(((group_ev_dones == True).sum(), 1), dtype=np.float32)
                all_eval_masks[group_id] = masks

                # save rewards
                if group_id == 0:
                    eval_episode_rewards_bad.append(group_ev_rewards)
                else:
                    eval_episode_rewards_good.append(group_ev_rewards)

            if all_args.save_gifs:
                image = single_env.render()
                all_frames.append(image)
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < all_args.ifi:
                    time.sleep(all_args.ifi - elapsed)
            
            
            if sum(eval_dones) == num_agents:
                success_step = step
                break
        
        eval_train_infos = []

        eval_episode_rewards_bad = np.array(eval_episode_rewards_bad)
        eval_episode_rewards_good = np.array(eval_episode_rewards_good)
        print("------------------------------------------------------------------------------------")
        print("episode:",ep)
        print("success step:", success_step)
        eval_average_episode_rewards_bad = np.mean(np.sum(np.array(eval_episode_rewards_bad), axis=0))
        eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards_bad})
        print("eval average episode rewards of group 0: "  + str(eval_average_episode_rewards_bad))

        eval_average_episode_rewards_good = np.mean(np.sum(np.array(eval_episode_rewards_good), axis=0))
        eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards_good})
        print("eval average episode rewards of group 1: "  + str(eval_average_episode_rewards_good))

        if all_args.save_gifs:
                imageio.mimsave(str(runner.gif_dir) + "/render_{}.gif".format(str(ep)), all_frames, duration=all_args.ifi)
    # post process
    envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])

