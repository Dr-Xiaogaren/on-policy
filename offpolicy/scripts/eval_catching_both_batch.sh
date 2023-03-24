#!/bin/sh
env="MPE"
scenario="simple_catching_expert_both"  # simple_speaker_listener # simple_reference
num_landmarks=0
num_agents=4
algo="maddpg"
exp="test_for_EnvV4_NoPreyVolo_full_1V1_SameAsMAPPO_withstopRW"
seed_max=1
model_dir="/home/zh/Documents/workspace/on-policy/offpolicy/scripts/results/MPE/simple_catching_expert_both/maddpg/EnvV4_NoPreyVolo_full_1V1_SameAsMAPPO_withstopRW/wandb/run-20230322_073929-sxrkyprz/files"
load_model_ep=-1
num_test_episode=300
step_mode="expert_prey" # assert mode == expert_adversary or  expert_both or  expert_prey or  none

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"


for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    echo "step_mode is ${step_mode}:"
    CUDA_VISIBLE_DEVICES=1 python eval/eval_catching_both_batch.py  --render_episodes 10  --use_wandb False --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 200 --num_env_steps 6000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --model_dir ${model_dir} --load_model_ep ${load_model_ep} --num_test_episode ${num_test_episode} --step_mode ${step_mode}
done