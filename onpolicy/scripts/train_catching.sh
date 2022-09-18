#!/bin/sh
env="MPE"
scenario="simple_catching"  # simple_speaker_listener # simple_reference
num_landmarks=0
num_agents=4
algo="rmappo"
exp0="EnvV0.1_TargetSpeed0.0_NewReward_AddPunish_ForExclusive_Linear_scale9"
exp1="EnvV0.1_TargetSpeed0.0_NewReward_AddPunish_ForExclusive_Linear_scale7"
exp2="EnvV0.1_TargetSpeed0.0_NewReward_AddPunish_ForExclusive_Linear_scale3"
seed_max=1
linear_punish_scale_0=9
linear_punish_scale_1=7
linear_punish_scale_2=3
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=0 python train/train_catching.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp0} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 24 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "joe-14807" --user_name "joe-14807" --linear_punish_scale ${linear_punish_scale_0};


echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=0 python train/train_catching.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp1} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 24 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "joe-14807" --user_name "joe-14807" --linear_punish_scale ${linear_punish_scale_1};


echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=0 python train/train_catching.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp2} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 24 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "joe-14807" --user_name "joe-14807" --linear_punish_scale ${linear_punish_scale_2};




