#!/bin/sh
env="MPE"
scenario="simple_catching"  # simple_speaker_listener # simple_reference
num_landmarks=0
num_agents=4
algo="rmappo"
exp0="check_TargetSpeed0.0_new_reward_addpunish_rwscale0.8"
exp1="check_TargetSpeed0.0_new_reward_addpunish_rwscale0.5"
exp2="check_TargetSpeed0.0_new_reward_addpunish_rwscale0.2"
seed_max=1
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "seed is ${seed}:"
echo "rw scale is 0.5"
CUDA_VISIBLE_DEVICES=0 python train/train_catching.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp0} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 24 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "joe-14807" --user_name "joe-14807" --rw_scale 0.5;

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "seed is ${seed}:"
echo "rw scale is 0.8"
CUDA_VISIBLE_DEVICES=0 python train/train_catching.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp1} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 24 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "joe-14807" --user_name "joe-14807" --rw_scale 0.8;

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "seed is ${seed}:"
echo "rw scale is 0.2"
CUDA_VISIBLE_DEVICES=0 python train/train_catching.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp2} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 24 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "joe-14807" --user_name "joe-14807" --rw_scale 0.2;
