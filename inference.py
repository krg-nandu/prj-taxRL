import tensorflow as tf
from baselines.common.models import build_impala_cnn
import procgen
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from baselines.common.policies import build_policy
from baselines.ppo2.runner import Runner
import numpy as np
import argparse, os
from pathlib import Path

NUM_EPISODES = 1000
NUM_ENVS = 512

# inference environment variables
distribution_mode = 'hard'

'''
    setting num_levels = 0 makes the program sample the full distribution of levels (INT32_MAX many). Although this
    technically can include the training data (levels 0-200), the chances of that happening are astronomically small,
    and the authors do it anyway ("we train and test agents on the full distribution of levels...").
'''
num_levels = 500 # 0

# runner and model variables
nsteps = 4000
nbatch_train = 0
max_grad_norm = 0.5
ent_coef = .01
vf_coef = 0.5
gamma = .999
lam = .95

class EvalRunner(Runner):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    def run(self):
        # Here, we init the lists that will contain the rewards at each time step for each env
        mb_rewards = []
        episode_reward_tallies = [0] * 100
        episode_rewards = []
        episodes = 0

        # For n in range number of steps
        #for _ in range(self.nsteps):
        while episodes <= NUM_EPISODES:
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            mb_rewards.append(rewards)
            for i, done in enumerate(self.dones):
                if done:
                    episode_reward = infos[i]['episode']['r']
                    episode_rewards.append(episode_reward)
                    print('played', episodes, 'episodes,       reward:', episode_reward, ',       average:', np.mean(episode_rewards))
                    episodes += 1
        
        #batch of steps to batch of rollouts
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_rewards = np.squeeze(np.sum(mb_rewards,0))

        return np.mean(episode_rewards)


def get_eval_env(env_name, num_envs):
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, distribution_mode=distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)

    return venv


def get_mean_reward(model, env_name, num_envs):
    env = get_eval_env(env_name, num_envs)
    mean_reward = evaluate(eval_env=env, model=model)

    return mean_reward


def load_model(model_path, env_name, num_envs):
    env = get_eval_env(env_name, num_envs)

    conv_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)
    network = conv_fn
    policy = build_policy(env, network)

    # Instantiate the model object (that creates act_model and train_model)
    from baselines.ppo2.model import Model
    model_fn = Model

    ob_space = env.observation_space
    ac_space = env.action_space
    nenvs = env.num_envs

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)

    load_path = model_path
    model.load(load_path)

    return model

def evaluate(*, eval_env, model):
    eval_runner = EvalRunner(env=eval_env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    mean_score = eval_runner.run()

    return mean_score


# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


if __name__ == "__main__":
    # recall that the episode buffer is biased towards shorter episodes -- so the first
    # samples may be of a very low or very high score in the case of climber (quick death or quick win)

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER" ]= "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    experiment_dir = 'train-procgen/train_procgen/results/new_experiments/{}'.format(args.experiment_name)

    if not os.path.isdir(experiment_dir):
        print('The input experiment location \'{}\' is not a directory.'.format(experiment_dir))
        exit()

    filepath = experiment_dir + '/' + 'manual_eval.csv'
    models_dir = experiment_dir + '/checkpoints'
    open(filepath, 'w').close() # clear the file
    file = open(filepath, 'a')
    file.write('model_checkpoint,meanreward\n')
    file.close()
    for model_checkpoint in sorted(os.listdir(models_dir)):
        model_checkpoint_path = models_dir + '/' + model_checkpoint
        print(model_checkpoint_path)
        if os.path.isdir(model_checkpoint_path):
            # skip directories
            continue
        with tf.Session(graph=tf.Graph()):
            file = open(filepath, 'a')
            model = load_model(model_checkpoint_path, args.env_name, num_envs=NUM_ENVS)

            checkpoint_number = int(model_checkpoint)
            average_reward = get_mean_reward(model, args.env_name, num_envs=NUM_ENVS)

            file.write('{},{}\n'.format(checkpoint_number, average_reward))
            file.close()
