#!/bin/sh
env="MPE"
scenario="simple_catching_expert_both"  # simple_speaker_listener # simple_reference
num_landmarks=0
num_agents=4
algo="rmappo"
exp="EnvV4_ExpPrey_NoPreyVolo_Addoffpolicy_SA_lowcriLR_ColidRW"
seed_max=1
step_mode="expert_prey"
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_catching_both.py --MAAC --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 16 --num_mini_batch 5 --episode_length 200 --num_env_steps 30000000 --ppo_epoch 4 --use_ReLU --gain 0.01 --lr 8e-6 --critic_lr 8e-6 --step_mode ${step_mode} --wandb_name "joe-14807" --user_name "joe-14807" ----data_chunk_length 12
done