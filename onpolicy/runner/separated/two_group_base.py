    
import time
from tokenize import group
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.util import update_linear_schedule

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.num_groups = 2
        self.num_goods = config['num_goods'] # 1
        self.num_bads = config['num_bads'] # 3
        self.obs_dict_keys = self.all_args.observation_dict

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)


        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy


        self.policy = []
        # 0 is bads , is goods
        for group_id in range(self.num_groups):
            share_observation_space = self.envs.share_observation_space[group_id] \
                                      if self.use_centralized_V else self.envs.observation_space[self.num_bads-1+group_id]
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[self.num_bads-1+group_id],
                        share_observation_space,
                        self.envs.action_space[self.num_bads-1+group_id],
                        device = self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore(self.all_args.load_model_ep)

        self.trainer = []
        self.buffer = []
        for group_id in range(self.num_groups):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[group_id], device = self.device)
            self.trainer.append(tr)

        for group_id in range(self.num_groups):
            # buffer
            share_observation_space = self.envs.share_observation_space[group_id] \
                                      if self.use_centralized_V else self.envs.observation_space[self.num_bads-1+group_id]
            num_inner_agent = self.num_bads if group_id == 0 else self.num_goods
            bu = SharedReplayBuffer(self.all_args,
                                    num_inner_agent,
                                    self.envs.observation_space[self.num_bads-1+group_id],
                                    share_observation_space,
                                    self.envs.action_space[self.num_bads-1+group_id]) 
            self.buffer.append(bu)


            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for group_id in range(self.num_groups):

            self.trainer[group_id].prep_rollout()
            # next_value = self.trainer[group_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
            #                                                     self.buffer[agent_id].rnn_states_critic[-1],
            #                                                     self.buffer[agent_id].masks[-1])
            # next_value = _t2n(next_value)
            # self.buffer[agent_id].compute_returns(next_value, self.trainer[group_id].value_normalizer)
            share_obs_input = dict()
            for key in self.obs_dict_keys:
                share_obs_input[key] = self.buffer[group_id].share_obs[key][-1]

            next_values = self.trainer[group_id].policy.get_values(share_obs_input,
                                                np.concatenate(self.buffer[group_id].rnn_states_critic[-1]),
                                                np.concatenate(self.buffer[group_id].masks[-1]))
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))

            self.buffer[group_id].compute_returns(next_values, self.trainer[group_id].value_normalizer)

    def train(self):
        train_infos = []
        for group_id in range(self.num_groups):
            self.trainer[group_id].prep_training()
            train_info = self.trainer[group_id].train(self.buffer[group_id])
            train_infos.append(train_info)       
            self.buffer[group_id].after_update()

        return train_infos

    def save(self, episode):
        for group_id in range(self.num_groups):
            policy_actor = self.trainer[group_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_group" + str(group_id) + "-ep" + str(episode) + ".pt")
            policy_critic = self.trainer[group_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_group" + str(group_id) + "-ep" + str(episode)+ ".pt")

    def restore(self, episode):
        for group_id in range(self.num_groups):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_group' + str(group_id) + "-ep" + str(episode) + '.pt')
            self.policy[group_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_group' + str(group_id) + "-ep" + str(episode) + '.pt')
            self.policy[group_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for group_id in range(self.num_groups):
            for k, v in train_infos[group_id].items():
                agent_k = "agent%i/" % group_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
