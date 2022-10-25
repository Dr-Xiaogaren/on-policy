# MAPPO Pursuit-Evasion Game

Chao Yu*, Akash Velu*, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. 

Website: https://sites.google.com/view/mappo

This repository implements MAPPO in the Pursuit-Evasion Game task. The implementation in this repositorory is used in the paper "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (https://arxiv.org/abs/2103.01955). 


## Environments supported:
- [Multiagent Particle-World Environments (MPEs)](https://github.com/openai/multiagent-particle-envs)

## 1. Usage

All core code is located within the onpolicy folder. The algorithms/ subfolder contains algorithm-specific code
for MAPPO and imitation learning. 

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
Copy map data from Server:
```
cd scene
scp -P 22 zh@180.201.0.53:/home/zh/Downloads/test4.png ./
scp -P 22 zh@180.201.0.53:/home/zh/Downloads/Goodyear_floor_trav_0_v3.png ./
scp -P 22 zh@180.201.0.53:/home/zh/Downloads/test3.png ./

```
The password of account `zh` is `123`.

## 3.Train
### 3.1 MAPPO
Here we use train_catching_both.sh as an example:
```
cd onpolicy/scripts
chmod +x ./train_catching_both.sh
./train_catching_both.sh
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 

### 3.2 Imitation learning
Similarly,
```
cd onpolicy/scripts
chmod +x ./imitation_train_catching_both.sh
./imitation_train_catching_both.sh
```
## 4.Evaluate
When testing, a folder containing two group models needs to be created:
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




