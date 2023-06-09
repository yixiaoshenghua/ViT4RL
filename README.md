# A simple yet work Vision Transformer for Sample Effcient online Reinforcement Learning

This repo is inherited from [CtrlFomer](https://arxiv.org/abs/2206.08883?context=cs.LG) (ICML2022) and is simplified to [single task reinforcement learning](https://github.com/YaoMarkMu/ViT4RL).

This is a PyTorch implementation of the contrastive reinforcement framework of **CtrlFomer** in single task reinforcement learning.

And for simplification, we replace the command line inputs from hydra-core to argparse. For logging, we replace wandb with logger class. We fix some bugs in original implementation and add some comments in timm.

## Requirements

We assume you have access to a gpu that can run CUDA 11.3 and Driver 460.91.03.

python 3.7
torch  1.9.1
gcc    9.2

Then, the simplest way to install all required dependencies is to create an anaconda environment by running


```

conda env create -f conda_env.yml

```

After the instalation ends you can activate your environment with
```

conda activate rl3

```



To train the ViT4RL run
```

bash scripts/run_walker_walk.sh

```


This will produce the `log` folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. To launch tensorboard run
```

tensorboard --logdir log

```

#### IMPORTANT: All the dropout operators in vision transformer are removed in this repo since it is not suit for online RL tasks. 

#### IMPORTANT: please use a batch size of 512 to reproduce the results in the paper.

#### IMPORTANT: And if action_repeat is used the effective number of env steps needs to be multiplied by action_repeat in the result graphs. This is a common practice for a fair comparison. The hyper-parameters of action_repeat for different task is set as follow:
|  Env   | action_repeat  |
|  ----  | ----  |
| cartpole_swingup  | 8 |
| reacher_easy  | 4 |
| cheetah_run  | 4 |
| finger_spin  | 2 |
| walker_walk  | 2 |
| ball_in_cup_catch  | 4 |


The console output is also available in a form:
```

| train | E: 5 | S: 5000 | R: 11.4359 | D: 66.8 s | BR: 0.0581 | ALOSS: -1.0640 | CLOSS: 0.0996 | TLOSS: -23.1683 | TVAL: 0.0945 | AENT: 3.8132

```
a training entry decodes as
```

train - training episode
E - total number of episodes
S - total number of environment steps
R - episode return
D - duration in seconds
BR - average reward of a sampled batch
ALOSS - average loss of the actor
CLOSS - average loss of the critic
TLOSS - average loss of the temperature parameter
TVAL - the value of temperature
AENT - the actor's entropy

```
while an evaluation entry
```

| eval | E: 20 | S: 20000 | R: 10.9356

```
contains
```

E - evaluation was performed after E episodes
S - evaluation was performed after S environment steps
R - average episode return computed over `num_eval_episodes` (usually 10)

```


## Acknowledgements
We used [timm](https://github.com/rwightman/pytorch-image-models) for basic model of vision transformer.
We used [kornia](https://github.com/kornia/kornia) for data augmentation.


## Reference
```
@inproceedings{mu2022ctrlformer,
  title={CtrlFormer: Learning Transferable State Representation for Visual Control via Transformer},
  author={Mu, Yao Mark and Chen, Shoufa and Ding, Mingyu and Chen, Jianyu and Chen, Runjian and Luo, Ping},
  booktitle={International Conference on Machine Learning},
  pages={16043--16061},
  year={2022},
  organization={PMLR}
}
```
