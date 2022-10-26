#!/bin/sh
env="MPE"
scenario="simple_catching_expert_both"  # simple_speaker_listener # simple_reference
num_landmarks=0
num_agents=4
algo="rmappo"
exp="EnvV2_TargetSpeed1.0_Egocentric_One-Mask_48_CheckCNN_Nocollide"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_catching_both.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 16 --num_mini_batch 10 --episode_length 200 --num_env_steps 20000000 --ppo_epoch 4 --use_ReLU --gain 0.01 --lr 3e-5 --critic_lr 3e-5 --wandb_name "joe-14807" --user_name "joe-14807"
done