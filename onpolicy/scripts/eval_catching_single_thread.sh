#!/bin/sh
env="MPE"
scenario="simple_catching"  # simple_speaker_listener # simple_reference
num_landmarks=0
num_agents=4
algo="rmappo"
exp="debug"
seed_max=1
model_dir="/workspace/on-policy/onpolicy/scripts/results/MPE/simple_catching/rmappo/check_TargetSpeed1.0_new_reward_addpunish_rwscale0.5/wandb/run-20220918_205140-1j5td93a/files"
load_model_ep=4100
num_test_episode=10
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python eval/eval_catching_single_thread.py --use_render True --render_episodes 1 --save_gifs True --use_wandb False --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 100 --num_env_steps 6000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --model_dir ${model_dir} --load_model_ep ${load_model_ep} --num_test_episode ${num_test_episode}
done