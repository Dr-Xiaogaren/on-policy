#!/bin/sh
env="MPE"
scenario="simple_catching_expert_both"  # simple_speaker_listener # simple_reference
num_landmarks=0
num_agents=4
algo="rmappo"
exp="test_for_EnvV4_NoExp_NoPreyVolo_full_1V1.2_SameAsMAPPO_withStopRW"
seed_max=1
maps_path='/home/zh/Documents/workspace/scene/val/hard'
model_dir="/home/zh/Documents/workspace/on-policy/onpolicy/scripts/results/MPE/simple_catching_expert_both/rmappo/EnvV4_ExpPrey_NoPreyVolo_full_1V1.2_SameAsMAPPO_withStopRW/wandb/run-20230330_184639-3kk7n03j/files"
load_model_ep=9300
num_test_episode=500
step_mode="expert_prey" # assert mode == expert_adversary or  expert_both or  expert_prey or  none
good_agent_speed=1.2
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"


for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    echo "step_mode is ${step_mode}:"
    CUDA_VISIBLE_DEVICES=1 python eval/eval_catching_both_batch.py  --render_episodes 10  --use_wandb False --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 200 --num_env_steps 6000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --model_dir ${model_dir} --load_model_ep ${load_model_ep} --num_test_episode ${num_test_episode} --step_mode ${step_mode} --maps_path ${maps_path} --good_agent_speed ${good_agent_speed} 
done