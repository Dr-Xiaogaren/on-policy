import torch
from onpolicy.algorithms.imitation_learning.algorithm.r_actor_critic import R_CNNActor
from onpolicy.utils.util import update_linear_schedule


class R_MAPPOPolicy_ForBC:
    """
    MAPPO Policy  class. Wraps actor networks to compute actions for behavior cloning.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.imitation_lr
        self.opti_eps = args.imitation_opti_eps
        self.weight_decay = args.imitation_weight_decay

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = R_CNNActor(args, self.obs_space, self.act_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
    
    def lr_decay(self, epoch, all_epoch):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, epoch, all_epoch, self.lr)

    def get_actions_probs(self, obs, rnn_states_actor, masks, available_actions=None):
        """
        Compute action probs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_probs: (torch.Tensor) probabilities of chosen actions.
        """
        action_probs = self.actor.get_probs(obs,
                                            rnn_states_actor,
                                            masks,
                                            available_actions
                                            )

        return action_probs

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
