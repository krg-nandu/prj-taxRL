import tensorflow as tf
import argparse, os, sys
sys.path.insert(0, './train-procgen/')

from baselines.common.models import build_impala_cnn, build_conv_ae
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
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_EPISODES = 512
NUM_ENVS = 64

distribution_mode = 'hard'
num_levels = 500

# runner and model variables
nsteps = 4096
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
        for _ in range(self.nsteps):
        #while episodes <= NUM_EPISODES:
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

            actions, values, self.states, neglogpacs, rloss, recon = self.model.step(self.obs, S=self.states, M=self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            mb_rewards.append(rewards)
            for i, done in enumerate(self.dones):
                if done:
                    episode_reward = infos[i]['episode']['r']
                    episode_rewards.append(episode_reward)
                    #print('played', episodes, 'episodes,       reward:', episode_reward, ',       average:', np.mean(episode_rewards))
                    episodes += 1
        
        #batch of steps to batch of rollouts
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_rewards = np.squeeze(np.sum(mb_rewards,0))

        print('Mean reward: {}'.format(np.mean(episode_rewards)))
        return np.mean(episode_rewards)


def get_eval_env(env_name, num_envs, vision_mode, stochasticity):
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, distribution_mode=distribution_mode, vision_mode=vision_mode, stochasticity=stochasticity)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)

    return venv


def get_mean_reward(model, env_name, num_envs, vision_mode=None, stochasticity=None):
    env = get_eval_env(env_name, num_envs, vision_mode, stochasticity)
    mean_reward = evaluate(eval_env=env, model=model)

    return mean_reward


def load_model(model_path, env_name, num_envs, vision_mode=None, stochasticity=None, use_decoder=False, bottleneck=None):
    env = get_eval_env(env_name, num_envs, vision_mode, stochasticity)

    if use_decoder:
        conv_fn = lambda x: build_conv_ae(x, bottleneck_dim=bottleneck, depths=[16,32,32], emb_size=256)
    else:
        conv_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)

    network = conv_fn
    policy = build_policy(env, network, use_decoder=use_decoder)

    # Instantiate the model object (that creates act_model and train_model)
    from baselines.ppo2.model import Model
    model_fn = Model

    ob_space = env.observation_space
    ac_space = env.action_space
    nenvs = env.num_envs

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, use_decoder=use_decoder)

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

''' USAGE
CUDA_VISIBLE_DEVICES=5 python eval_model.py --eval_env_name heist --env_name heistp --experiment_name heist_p75 --use_decoder --bottleneck 256 --BASE_PATH /media/data_cifs/projects/prj_procgen/lax/procgen-p7/procgen/train-procgen/train_procgen/results
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--BASE_PATH', type=str, required=True)
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--vision_mode', type=str, default='normal', choices=["normal", "semantic_mask", "fg_mask"], required=True)
    parser.add_argument('--stochasticity', type=float, default=1.)

    parser.add_argument('--eval_env_name', type=str)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--use_decoder', action='store_true')
    parser.add_argument('--bottleneck', type=int, default=0)
    parser.add_argument('--output', type=str, required=True)
 
    args = parser.parse_args()

    BASE_PATH = args.BASE_PATH
    experiment_dir = os.path.join(BASE_PATH, args.env_name, args.experiment_name)
    if not os.path.isdir(experiment_dir):
        print('The input experiment location \'{}\' is not a directory.'.format(experiment_dir))
        os._exit(0)

    filepath = os.path.join(experiment_dir, '{}.csv'.format(args.output))
    models_dir = os.path.join(experiment_dir, 'checkpoints')

    open(filepath, 'w').close() # clear the file
    file = open(filepath, 'a')
    file.write('model_checkpoint,meanreward\n')
    file.close()

    for model_checkpoint in sorted(os.listdir(models_dir)):
        model_checkpoint_path = os.path.join(models_dir, model_checkpoint)
        print(model_checkpoint_path)
        if os.path.isdir(model_checkpoint_path):
            # skip directories
            continue
        with tf.Session(graph=tf.Graph()):
            file = open(filepath, 'a')

            my_env = args.env_name
            if args.eval_env_name != None:
                my_env = args.eval_env_name
            
            model = load_model(model_checkpoint_path, my_env, vision_mode=args.vision_mode, stochasticity=args.stochasticity, num_envs=NUM_ENVS, use_decoder=args.use_decoder, bottleneck=args.bottleneck)
 
            checkpoint_number = int(model_checkpoint)
            average_reward = get_mean_reward(model, my_env, num_envs=NUM_ENVS, vision_mode=args.vision_mode, stochasticity=args.stochasticity)

            file.write('{},{}\n'.format(checkpoint_number, average_reward))
            file.close()
