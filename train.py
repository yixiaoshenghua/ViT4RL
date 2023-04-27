import copy
import math
import os
import pickle as pkl
import sys
import time

import numpy as np
import json
import dmc2gym
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder
from drq import DRQAgent

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'

torch.backends.cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser(description='ViT4RL')
    # env
    parser.add_argument("--env", default='cartpole_swingup', type=str)
    parser.add_argument("--env_index", default=0, type=int)
    parser.add_argument("--token_index", default=0, type=int)
    parser.add_argument("--action_repeat", default=2, type=int)
    # train
    parser.add_argument("--num_train_steps", default=100000, type=int)
    parser.add_argument("--num_train_iters", default=1, type=int)
    parser.add_argument("--num_seed_steps", default=1000, type=int)
    parser.add_argument("--replay_buffer_capacity", default=100000, type=int)
    parser.add_argument("--seed", default=1, type=int)
    # eval
    parser.add_argument("--eval_frequency", default=5000, type=int)
    parser.add_argument("--num_eval_episodes", default=10, type=int)
    # misc
    parser.add_argument("--log_dir", default='log', type=str)
    parser.add_argument("--log_frequency_step", default=10000, type=int)
    parser.add_argument("--log_save_tb", default=False, action="store_true")
    parser.add_argument("--save_video", default=False, action="store_true")
    parser.add_argument("--save_model", default=False, action="store_true")
    parser.add_argument("--save_model_frequency", default=50000, type=int)
    parser.add_argument("--device", default=-1)
    # observation
    parser.add_argument("--image_size", default=84, type=int)
    parser.add_argument("--image_pad", default=4, type=int)
    parser.add_argument("--frame_stack", default=3, type=int)
    # global params
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--encoder_conf", default=0.0025)
    parser.add_argument("--load_pretrain", default=False, action="store_true")
    parser.add_argument("--scale", default=0.0425)
    parser.add_argument("--tag", default='trans_same')
    # agent configuration
    parser.add_argument("--agent_name", default='drq')
    parser.add_argument("--discount", default=0.99)
    parser.add_argument("--init_temperature", default=0.1)
    parser.add_argument("--actor_update_frequency", default=2, type=int)
    parser.add_argument("--critic_tau", default=0.01)
    parser.add_argument("--critic_target_update_frequency", default=2, type=int)
    # critic
    parser.add_argument("--hidden_dim", default=1024, type=int)
    parser.add_argument("--hidden_depth", default=3, type=int)
    # actor
    parser.add_argument("--log_std_max", default=2, type=int)
    parser.add_argument("--log_std_min", default=-10, type=int)
    # encoder
    parser.add_argument("--feature_dim", default=50, type=int)
    parser.add_argument("--encoder_cfg", default=0, type=int)
    args = parser.parse_args()
    return args

def make_agent(args):
    agent = DRQAgent(
        args,
        obs_shape=args.obs_shape,
        action_shape=args.action_shape,
        action_range=args.action_range,
        device=args.device,
        discount=args.discount,
        init_temperature=args.init_temperature,
        lr=args.lr,
        actor_update_frequency=args.actor_update_frequency,
        critic_tau=args.critic_tau,
        critic_target_update_frequency=args.critic_target_update_frequency,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
        encoder_cfg=args.encoder_cfg
    )
    return agent

def make_env(args):
    """Helper function to create dm_control environment"""
    if args.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif args.env == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    else:
        domain_name = args.env.split('_')[0]
        task_name = '_'.join(args.env.split('_')[1:])

    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'quadruped' else 0

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=args.seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=args.image_size,
                       width=args.image_size,
                       frame_skip=args.action_repeat,
                       camera_id=camera_id)

    env = utils.FrameStack(env, k=args.frame_stack)

    env.seed(args.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, args):
        self.work_dir = os.getcwd() + '/' + args.log_dir
        # make directory
        ts = time.gmtime()
        ts = time.strftime("%m-%d-%H-%M", ts)
        env_name = args.env
        exp_name = env_name + '-' + ts +'-b' + str(args.batch_size) + '-s' + str(args.seed)
        self.work_dir = self.work_dir + '/' + exp_name
        print(f'workspace: {self.work_dir}')
        self.args = args

        self.logger = Logger(self.work_dir,
                             save_tb=args.log_save_tb,
                             log_frequency=args.log_frequency_step,
                             agent=args.agent_name,
                             action_repeat=args.action_repeat)
        
        if self.args.save_model and not os.path.exists(os.path.join(self.work_dir, 'models')):
            os.mkdir(os.path.join(self.work_dir, 'models'))

        with open(os.path.join(self.work_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        args.device = 'cuda:'+str(utils.get_device(12000)) if args.device == -1 and torch.cuda.is_available() else 'cpu'
        utils.set_seed_everywhere(args.seed)
        self.device = torch.device(args.device)
        self.env = make_env(args)

        args.obs_shape = self.env.observation_space.shape
        args.action_shape = self.env.action_space.shape
        args.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = make_agent(args)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          args.replay_buffer_capacity,
                                          self.args.image_pad, self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if args.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.args.num_eval_episodes):
            obs = self.env.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.args.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        load_path="./models"
        print("start load")
        self.agent.actor.load_state_dict(torch.load(load_path+"/actor_stand_model.pth"))
        self.agent.critic.load_state_dict(torch.load(load_path+"/critic_stand_model.pth"))
        self.agent.critic_target.load_state_dict(torch.load(load_path+"/critic_stand_model.pth"))
        print("load pretrained model done")
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()

        while self.step < self.args.num_train_steps:

            # evaluate agent periodically
            if (self.step+1) % self.args.eval_frequency == 0:
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate()

            if (self.step+1) % self.args.save_model_frequency == 0 and self.save_model:
                self.agent.save_model(self.work_dir+"/models", self.step+1)

            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.args.num_seed_steps))

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.args.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.args.num_seed_steps:
                for _ in range(self.args.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


def main(args):
    from train import Workspace as W
    workspace = W(args)
    workspace.run()


if __name__ == '__main__':
    args = get_args()
    main(args)
