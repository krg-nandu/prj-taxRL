import os, argparse, sys
import tensorflow as tf
sys.path.insert(0, '../baselines')
from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
sys.path.insert(1, '../../procgen')
from procgen import ProcgenEnv

def train_fn(env_name, num_envs, distribution_mode, num_levels, start_level, timesteps_per_proc, is_test_worker=False,
                log_dir='/tmp/procgen', comm=None, experiment_name='temp', num_episodes=0, get_test_perf=False, learning_rate=5e-4,
                vision_mode='normal', checkpoint_interval=50):
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else num_levels

    if log_dir is not None:
        log_comm = comm.Split(1 if is_test_worker else 0, 0)
        format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
        logger.configure(comm=log_comm, dir=log_dir, format_strs=format_strs)

    logger.info("Creating environment(s)...")
    def make_env(is_eval_env):
        venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=0 if is_eval_env else num_levels, start_level=start_level,
            distribution_mode=distribution_mode, vision_mode=vision_mode)
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False)
        return venv
    venv = make_env(is_eval_env=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    logger.info("training")
    ppo2.learn(
        env=venv,
        eval_env=make_env(is_eval_env=True) if get_test_perf else None,
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        save_interval=checkpoint_interval,
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=1,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=use_vf_clipping,
        comm=comm,
        lr=learning_rate,
        cliprange=clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
        env_name=env_name,
        experiment_name=experiment_name,
        num_levels=num_levels,
        num_episodes=num_episodes,
        distribution_mode=distribution_mode,
        vision_mode=vision_mode,
    )

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    # Required
    parser.add_argument('--game', type=str, required=True)
    parser.add_argument('--vision_mode', type=str, required=True, choices=['normal', 'semantic_mask', 'fg_mask'])
    # Not required
    parser.add_argument('--experiment_name', type=str, default='temp')
    parser.add_argument('--batch_size', type=int, default=256, choices=[64, 128, 256, 512]) # synonymous with "num_envs"
    parser.add_argument('--episodes_logged_per_batch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--get_test_perf', type=bool, default=False)
    parser.add_argument('--log_dir_root', type=str, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=50)
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=500)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    args = parser.parse_args()
    
    # assuming you are using 1 GPU, this generalizes the relationship between timesteps_per_proc and num_envs=batch_size (per OpenAI)
    timesteps_per_proc = 50_000_000 * (args.batch_size / 64) 
    root = args.log_dir_root if args.log_dir_root is not None else 'train_procgen/results/new_experiments'
    log_dir = '{}/{}'.format(root, args.game + '/' + args.experiment_name if args.experiment_name is not 'temp' else 'tmp')
    if os.path.isdir(log_dir) and args.experiment_name is not 'temp':
        print('Error: {}/ is already a directory. Please use a unique --experiment_name.'.format(log_dir))
        exit()
    else:
        os.makedirs(log_dir, exist_ok=True)

    ### Currently unused when using just 1 GPU
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)
    ###

    train_fn(args.game,
        args.batch_size,
        args.distribution_mode,
        args.num_levels,
        args.start_level,
        timesteps_per_proc,
        is_test_worker=is_test_worker,
        log_dir=log_dir,
        comm=comm,
        experiment_name=args.experiment_name,
        num_episodes=args.episodes_logged_per_batch,
        get_test_perf=args.get_test_perf,
        learning_rate=args.learning_rate,
        vision_mode=args.vision_mode,
        checkpoint_interval=args.checkpoint_interval,
    )

if __name__ == '__main__':
    main()
