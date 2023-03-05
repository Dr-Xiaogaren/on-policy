import numpy as np
import torch
import torch.nn as nn
from offpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from offpolicy.utils.valuenorm import ValueNorm
from offpolicy.algorithms.utils.util import check

class R_MAAC():
    """
    Trainer class for MADDPG to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.num_update_each = args.num_update_each
        self.batch_size = args.batch_size
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.gamma = args.gamma

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        self.use_soft_update = args.use_soft_update
             

    def maddpg_update(self, sample):
        q_loss, critic_grad_norm = self.update_critic(sample)
        policy_loss, action_probs, actor_grad_norm = self.update_policy(sample)
        # q_loss, critic_grad_norm, policy_loss, action_probs, actor_grad_norm = self.update_together(sample)
        
        entropy = torch.sum(-torch.log(action_probs)*action_probs, dim=-1).mean()

        return q_loss, policy_loss, entropy, critic_grad_norm, actor_grad_norm

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0

        for _ in range(self.num_update_each):
            if self._use_recurrent_policy:
                sample = buffer.recurrent_generator(self.batch_size, self.data_chunk_length)

                q_loss, policy_loss, entropy, critic_grad_norm, actor_grad_norm = self.maddpg_update(sample)

                train_info['value_loss'] += q_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['entropy'] += entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
        
        if self.use_soft_update:
            self.policy.soft_target_updates()
        else:
            self.policy.hard_target_updates()

        num_updates = self.num_update_each

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.target_actor.train()
        self.policy.target_critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()        
        self.policy.target_actor.eval()
        self.policy.target_critic.eval()

    def update_critic(self, sample):

        # shape:(n_robots,data_chunk_length,batch_size,...)
        # share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        # value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        # adv_targ, available_actions_batch = sample

        obs_batch, rnn_states_batch, rnn_states_critic_batch, masks_batch, \
        next_obs_batch, next_rnn_states_batch, next_rnn_states_critic_batch, next_mask_batch, \
        actions_batch, return_batch  = sample

        return_batch = check(return_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        next_mask_batch = check(next_mask_batch).to(**self.tpdv)
        
        # Sacrifice a value for convenience
        for key in obs_batch.keys():
            obs_batch[key] = check(obs_batch[key]).to(**self.tpdv) 
            next_obs_batch[key] = check(next_obs_batch[key]).to(**self.tpdv) 


        # calculate next Q value
        next_Q,_ ,_ ,_ ,_ = self.policy.get_actions(next_obs_batch, next_rnn_states_batch, next_rnn_states_critic_batch, 
                                            next_mask_batch, use_target_actor=True, use_target_critic=True)
        # calculate current Q value
        critic_rets= self.policy.get_values(obs_batch, actions_batch, rnn_states_critic_batch, masks_batch, use_target=False)
        # calculate target Q value
        target_q = return_batch.view(-1,1) + self.gamma*next_Q.view(-1,1)*next_mask_batch.view(-1,1)
        
        q_loss =  torch.nn.MSELoss()(critic_rets,target_q.detach())

        self.policy.critic_optimizer.zero_grad()

        q_loss.backward()

        critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return q_loss, critic_grad_norm

    def update_policy(self, sample):
        obs_batch, rnn_states_batch, rnn_states_critic_batch, masks_batch, \
        next_obs_batch, next_rnn_states_batch, next_rnn_states_critic_batch, next_mask_batch, \
        actions_batch, return_batch  = sample

        curr_q, all_q, action_probs, log_action_prob= self.policy.evaluate_actions(obs_batch, rnn_states_batch, rnn_states_critic_batch, 
                                                                    masks_batch, use_target_actor=False, use_target_critic=False)
        
        v = (all_q*action_probs).sum(dim=1,keepdim=True)
        pol_target = curr_q - v
        policy_loss = (log_action_prob*(-pol_target).detach()).mean()
        # policy_loss += (action_probs**2).mean()*1e-3

        self.policy.actor_optimizer.zero_grad()      
        for p in self.policy.critic.parameters():
            p.requires_grad = False

        policy_loss.backward()
        
        for p in self.policy.critic.parameters():
            p.requires_grad = True

        actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        return policy_loss, action_probs, actor_grad_norm


    def update_together(self, sample):
        obs_batch, rnn_states_batch, rnn_states_critic_batch, masks_batch, \
        next_obs_batch, next_rnn_states_batch, next_rnn_states_critic_batch, next_mask_batch, \
        actions_batch, return_batch  = sample

        return_batch = check(return_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        
        # Sacrifice a value for convenience
        for key in obs_batch.keys():
            obs_batch[key] = check(obs_batch[key]).to(**self.tpdv) 
            next_obs_batch[key] = check(next_obs_batch[key]).to(**self.tpdv) 


        # calculate next Q value
        next_Q,_ ,_ ,_ ,_ = self.policy.get_actions(next_obs_batch, next_rnn_states_batch, next_rnn_states_critic_batch, 
                                            next_mask_batch, use_target_actor=True, use_target_critic=True)
        # calculate current Q value
        critic_rets= self.policy.get_values(obs_batch, actions_batch, rnn_states_critic_batch, masks_batch, use_target=False)
        # calculate target Q value
        target_q = return_batch.view(-1,1) + self.gamma*next_Q.view(-1,1)*next_mask_batch.view(-1,1)
        
        q_loss =  torch.nn.MSELoss()(critic_rets,target_q.detach())

        self.policy.critic_optimizer.zero_grad()
        self.policy.actor_optimizer.zero_grad()


        curr_q, all_q, action_probs, log_action_prob= self.policy.evaluate_actions(obs_batch, rnn_states_batch, rnn_states_critic_batch, 
                                                                    masks_batch, use_target_actor=False, use_target_critic=False)
        
        v = (all_q*action_probs).sum(dim=1,keepdim=True)
        pol_target = curr_q - v
        policy_loss = (log_action_prob*(-pol_target.detach())).mean()
        policy_loss += (action_probs**2).mean()*1e-3

        q_loss.backward()
        policy_loss.backward()

        actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.actor_optimizer.step()
        self.policy.critic_optimizer.step()

        return q_loss, critic_grad_norm, policy_loss, action_probs, actor_grad_norm














        









        

            




