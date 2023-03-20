# MAPPO Pursuit-Evasion Game

Hao zhang.

## Environments supported:
- [Multiagent Particle-World Environments (MPEs)](https://github.com/openai/multiagent-particle-envs)

## 1. Usage

We reproduced algorithms of MAGGPD, MAAC, MAPPO, and IPPO. All core code is located within the onpolicy and offpolicy folder. The algorithms/ subfolder contains algorithm-specific code
for each algorithm. 

* The envs/ subfolder contains environment wrapper implementations for the MPEs

* Code to perform training rollouts and policy updates are contained within the runner/ folder - there is a runner for 
each environment. 

* Executable scripts for training with default hyperparameters can be found in the scripts/ folder. The files are named
in the following manner: train_algo_environment.sh. Within each file, the map name (in the case of SMAC and the MPEs) can be altered. 
* Python training scripts for each environment can be found in the scripts/train/ folder. 

* The config.py file contains relevant hyperparameter and env settings. Most hyperparameters are defaulted to the ones
used in the paper; however, please refer to the appendix for a full list of hyperparameters used. 


## 2. Installation & Environment
### 2.1 Using Docker
```
sudo docker pull laughingjoe/mappo:light
```

```
docker run -p 127.0.0.1:8888:8888 --gpus all -i -t \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=$DISPLAY \
-w / --ipc=host \
laughingjoe/mappo:origin \
/bin/zsh
```

```
conda activate marl
```
If there is no map data, you need to create a corresponding folder：
```
cd on-policy
mkdir scene
```
Copy map data from Server to the scene directory.


## 3.Train
### 3.1 MAPPO
Here we use train_catching_both_mappo.sh as an example:
```
cd onpolicy/scripts
chmod +x ./train_catching_both_mappo.sh
./train_catching_both_mappo.sh
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 

### 3.2 IPPO
```
cd onpolicy/scripts
chmod +x ./train_catching_both_ippo.sh
./train_catching_both_ippo.sh
```
### 3.3 MADDPG

```
cd offpolicy/scripts
chmod +x ./train_catching_both_maddpg.sh
./train_catching_both_maddpg.sh
```
### 3.4 MAAC

```
cd offpolicy/scripts
chmod +x ./train_catching_both_MAAC.sh
./train_catching_both_MAAC.sh
```
### 3.5 Some Notes for Train
* When setting the number of robots in each group, the total number of robots needs to be adjusted at the same time.
* To set the speed of the robot, one needs to modify it directly in the `mpe/scenarios/simple_catching_expert_both.py` file.
* Each group comes with a heuristic that needs to be set with the --step_mode parameter. The optional range is (***expert_adversary***, ***expert_both***, ***expert_prey***, ***none***). ***expert_adversary*** and ***expert_prey*** mean that only the hunter or prey use the heuristic respectively. ***expert_both*** and ***none*** mean that both use or neither use heuristics.

## 4.Evaluate
Here we use MAPPO as an example. When testing, a folder containing two group models needs to be created:
```
cd on-policy
mkdir data
```
Copy pretrained model from Server:
```

cd data
scp -P 22 zh@180.201.0.53:/home/zh/Downloads/actor_group1-ep900.pt ./
scp -P 22 zh@180.201.0.53:/home/zh/Downloads/actor_group0-ep900.pt ./
```
Then in `eval_catching_both.sh`, set the `--model_dir` to the data directory and `--load_model_ep` to 900(same as the ep number in the name of pretrained models). If visualization is not required, remember to remove `-- save_gifs`. Next, just run:
```
cd onpolicy/scripts
chmod +x eval_catching_both.sh
eval_catching_both.sh
```
The visualization function has been modified to `scripts/render_catching_both.sh`

## 5. Acknowledgement
Our code frameworks are mainly implemented based on [MAPPO](https://github.com/marlbenchmark/on-policy.) benchmark.