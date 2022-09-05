    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.two_group_base import Runner
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for group_id in range(self.num_groups):
                    self.trainer[group_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    for group_id in range(self.num_groups):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in infos[group_id].keys():
                                idv_rews.append(infos[group_id]['individual_reward'])
                        train_infos[group_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[group_id].update({"average_episode_rewards": np.mean(self.buffer[group_id].rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        for group_id in range(self.num_groups):
            group_obs = obs[:,:self.num_bads] if group_id == 0 else obs[:,self.num_bads:]
            num_inner_agent = self.num_bads if group_id == 0 else self.num_goods
            if self.use_centralized_V:
                share_obs = group_obs.reshape(self.n_rollout_threads, -1)
                share_obs = np.expand_dims(share_obs, 1).repeat(num_inner_agent, axis=1)
            else:
                share_obs = group_obs
  
            self.buffer[group_id].share_obs[0] = share_obs.copy()
            self.buffer[group_id].obs[0] = group_obs.copy()



    @torch.no_grad()
    def collect(self, step):
        all_values = []
        all_actions = []
        all_temp_actions_env = []
        all_action_log_probs = []
        all_rnn_states = []
        all_rnn_states_critic = []

        for group_id in range(self.num_groups):
            self.trainer[group_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[group_id].policy.get_actions(np.concatenate(self.buffer[group_id].share_obs[step]),
                            np.concatenate(self.buffer[group_id].obs[step]),
                            np.concatenate(self.buffer[group_id].rnn_states[step]),
                            np.concatenate(self.buffer[group_id].rnn_states_critic[step]),
                            np.concatenate(self.buffer[group_id].masks[step]))
            # [agents, envs, dim]

            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
            action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
            # rearrange action
            if self.envs.action_space[self.num_bads-1+group_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[self.num_bads-1+group_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[self.num_bads-1+group_id].high[i]+1)[actions[:, :, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=2)
            elif self.envs.action_space[self.num_bads-1+group_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[self.num_bads-1+group_id].n)[action], 2)
            else:
                raise NotImplementedError
            
            all_values.append(values)
            all_actions.append(action)
            all_temp_actions_env.append(action_env)
            all_action_log_probs.append(action_log_prob)
            all_rnn_states.append(rnn_state)
            all_rnn_states_critic.append(rnn_state_critic)

        actions_env = np.concatenate(all_temp_actions_env, axis=1)

        return all_values, all_actions, all_action_log_probs, all_rnn_states, all_rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, all_values, all_actions, all_action_log_probs, all_rnn_states, all_rnn_states_critic = data
        for group_id in(self.num_groups):
            # devide obs, reward, and dones into two groups
            group_obs = obs[:,:self.num_bads] if group_id == 0 else obs[:,self.num_bads:]
            group_dones = dones[:,:self.num_bads] if group_id == 0 else dones[:,self.num_bads:]
            group_rewards = rewards[:,:self.num_bads] if group_id == 0 else rewards[:,self.num_bads:]
            group_infos = infos[:,:self.num_bads] if group_id == 0 else infos[:,self.num_bads:]

            all_rnn_states[group_id][group_dones == True] = np.zeros(((group_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            all_rnn_states_critic[group_id][group_dones == True] = np.zeros(((group_dones == True).sum(), *self.buffer[group_id].rnn_states_critic.shape[3:]), dtype=np.float32)
            num_inner_agent = self.num_bads if group_id == 0 else self.num_goods
            masks = np.ones((self.n_rollout_threads, num_inner_agent, 1), dtype=np.float32)
            masks[group_dones == True] = np.zeros(((group_dones == True).sum(), 1), dtype=np.float32)

            if self.use_centralized_V:
                share_obs = group_obs.reshape(self.n_rollout_threads, -1)
                share_obs = np.expand_dims(share_obs, 1).repeat(num_inner_agent, axis=1)
            else:
                share_obs = group_obs
            
            self.buffer[group_id].insert(share_obs, group_obs, all_rnn_states[group_id], all_rnn_states_critic[group_id],
                               all_actions[group_id], all_action_log_probs[group_id], all_values[group_id], group_rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        # two group
        eval_episode_rewards_good = []
        eval_episode_rewards_bad = []
        eval_obs = self.eval_envs.reset()
        # two groups
        all_eval_rnn_states = [np.zeros((self.n_eval_rollout_threads, *self.buffer[0].rnn_states.shape[2:]), dtype=np.float32),
                               np.zeros((self.n_eval_rollout_threads, *self.buffer[1].rnn_states.shape[2:]), dtype=np.float32)]
        all_eval_masks = [np.ones((self.n_eval_rollout_threads, self.num_bads, 1), dtype=np.float32),
                          np.ones((self.n_eval_rollout_threads, self.num_goods, 1), dtype=np.float32)]

        for eval_step in range(self.episode_length):
            all_eval_actions_env = []
            for group_id in range(self.num_groups):
                self.trainer[group_id].prep_rollout()
                group_obs = eval_obs[:,:self.num_bads] if group_id == 0 else eval_obs[:,self.num_bads:]

                eval_action, eval_rnn_states = self.trainer[group_id].policy.act(np.concatenate(group_obs),
                                                np.concatenate(all_eval_rnn_states[group_id]),
                                                np.concatenate(all_eval_masks[group_id]),
                                                deterministic=True)
                
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

                if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[0].shape):
                        eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                        if i == 0:
                            eval_actions_env = eval_uc_actions_env
                        else:
                            eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
                elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                    eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
                else:
                    raise NotImplementedError
            
                all_eval_actions_env.append(eval_actions_env)
            
            # gather  all groups' action into one array
            actions_env = np.concatenate(all_eval_actions_env, axis=1)
            # evaluate 
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(actions_env)


            for group_id in range(self.num_groups):
                # split into two groups
                group_ev_rewards = eval_rewards[:,:self.num_bads] if group_id == 0 else eval_rewards[:,self.num_bads:]
                group_ev_dones = eval_dones[:,:self.num_bads] if group_id == 0 else eval_dones[:,self.num_bads:]

                all_eval_rnn_states[group_id][group_ev_dones == True] = np.zeros(((group_ev_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                
                num_inner_agent = self.num_bads if group_id == 0 else self.num_goods
                masks = np.ones((self.n_rollout_threads, num_inner_agent, 1), dtype=np.float32)
                masks[group_ev_dones == True] = np.zeros(((group_ev_dones == True).sum(), 1), dtype=np.float32)
                all_eval_masks[group_id] = masks

                # save rewards
                if group_id == 0:
                    eval_episode_rewards_bad.append(group_ev_rewards)
                else:
                    eval_episode_rewards_good.append(group_ev_rewards)
                
            eval_train_infos = []

            eval_episode_rewards_bad = np.array(eval_episode_rewards_bad)
            eval_episode_rewards_good = np.array(eval_episode_rewards_good)

            eval_average_episode_rewards_bad = np.mean(np.sum(np.array(eval_episode_rewards_bad), axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards_bad})
            print("eval average episode rewards of group 0: "  + str(eval_average_episode_rewards_bad))

            eval_average_episode_rewards_good = np.mean(np.sum(np.array(eval_episode_rewards_good), axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards_good})
            print("eval average episode rewards of group 1: "  + str(eval_average_episode_rewards_good))

            self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
