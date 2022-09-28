#!/bin/sh
env="MPE"
scenario="simple_catching_expert_prey"  # simple_speaker_listener # simple_reference
num_landmarks=0
num_agents=4
algo="rmappo"
exp="TargetSpeed1.0_new_reward_addpunish_rwscale0.5_Expert"
seed_max=1
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "seed is ${seed}:"
echo "rw scale is 0.5"
CUDA_VISIBLE_DEVICES=0 python train/train_catching_expert_prey.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 24 --num_mini_batch 1 --episode_length 200 --num_env_steps 20000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "joe-14807" --user_name "joe-14807" --rw_scale 0.5;

