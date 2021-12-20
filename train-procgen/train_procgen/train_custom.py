#import torch
#torch.cuda.current_device()

import os, argparse, sys
sys.path.insert(0, './train-procgen/')
import tensorflow as tf
from baselines.ppo2 import ppo2
#from baselines.ppo2 import ppo2_custom as ppo2
from baselines.common.models import build_impala_cnn, build_conv_ae

#from baselines.common.mpi_util import setup_mpi_gpus
#from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI

sys.path.insert(1, './')
from procgen import ProcgenEnv

#from models.unet_model import UNet

save_interval = 50

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train_fn(
        env_name, 
        num_envs, 
        distribution_mode, 
        num_levels, 
        start_level, 
        timesteps_per_proc, 
        is_test_worker=False,
        log_dir='/tmp/procgen', 
        comm=None, 
        experiment_name=None, 
        num_episodes=0, 
        test_eval=False, 
        test_model_segmented=False, 
        ae_coeff=0., 
        bottleneck=None,
        load_path = None, 
        save_recon_gif = False, 
        exp_name='testgif', 
        use_decoder=None, 
        stochasticity=None,
        vision_mode=None):

    learning_rate = 5e-4
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

    logger.info("creating environment")
    venv = ProcgenEnv(
        num_envs=num_envs, 
        env_name=env_name, 
        num_levels=num_levels, 
        start_level=start_level,
        distribution_mode=distribution_mode,
        stochasticity=stochasticity,
        vision_mode=vision_mode
    )

    eval_venv = ProcgenEnv(
            num_envs=num_envs, 
            env_name=env_name, 
            num_levels=0, 
            start_level=start_level,
            distribution_mode=distribution_mode,
            stochasticity=stochasticity,
            vision_mode=vision_mode
    )

    venv = VecExtractDictObs(venv, "rgb")
    eval_venv = VecExtractDictObs(eval_venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )
    eval_venv = VecMonitor(
        venv=eval_venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)
    eval_venv = VecNormalize(venv=eval_venv, ob=False)

    logger.info("creating tf session")
    #setup_mpi_gpus()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    if use_decoder:
        conv_fn = lambda x: build_conv_ae(x, bottleneck_dim=bottleneck, depths=[16,32,32], emb_size=256)
    else:
        conv_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)

    net = None
    if test_model_segmented and env_name == 'climber':
        # this so far works only for climber!
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cuda:1'
        ckpt_path = '/media/data_cifs/projects/prj_procgen/sem-seg-rl/ckpts/unet_v0/ckpt_epoch_48_loss_9.431936632608995e-05.pth'
        net = UNet(n_channels=3, n_classes=5, bilinear=True) 
        net = net.to(device=device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device))
        net = net.eval()

    logger.info("training")

    ppo2.learn(
        env=venv,
        eval_env=eval_venv if test_eval else None,
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        save_interval=save_interval,
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
        seg_net = net,
        load_path = load_path,
        ae_coeff = ae_coeff,
        save_recon_gif = save_recon_gif,
        exp_name = exp_name,
        use_decoder = use_decoder
    )

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')

    ## TRAINING PARAMS
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--num_envs', type=int, required=True)
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"], required=True)
    parser.add_argument('--num_levels', type=int, required=True)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--timesteps_per_proc', type=int, default=50_000_000)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=0)
    parser.add_argument('--test_eval', type=bool, default=False)

    ## VISUAL REP
    parser.add_argument('--vision_mode', type=str, default='normal', choices=["normal", "semantic_mask", "fg_mask"], required=True)

    ## STOCHASTIC REWARDS
    parser.add_argument('--stochasticity', type=float, default=1.)

    ## TEST WITH SEG-MODEL IN PLACE
    parser.add_argument('--test_model_segmented', action='store_true')

    ## AUTO ENCODER HYPERPARAMS
    parser.add_argument('--use_decoder', action='store_true')
    parser.add_argument('--bottleneck', type=int, default=0)
    parser.add_argument('--ae_coeff', type=float, default=0.)

    ## EVAL/VIS HYPERPARAMS
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_recon_gif', action='store_true')
    parser.add_argument('--gif_name', type=str, default='testgif')

    args = parser.parse_args()

    abspath = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(abspath)

    log_dir = os.path.join(
                    current_dir_path,
                    'results',
                    args.env_name,
                    args.experiment_name
                    )

    if os.path.isdir(log_dir):
        print('Error: {}/ is already a directory, try a different --experiment_name.'.format(log_dir))
        os._exit(0)
    else:
        os.makedirs(log_dir)
        print('logging results to {}/.'.format(log_dir))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)

    train_fn(args.env_name,
        args.num_envs,
        args.distribution_mode,
        args.num_levels,
        args.start_level,
        args.timesteps_per_proc,
        is_test_worker=is_test_worker,
        log_dir=log_dir,
        comm=comm,
        experiment_name=args.experiment_name,
        num_episodes=args.num_episodes,
        test_eval=args.test_eval,
        test_model_segmented = args.test_model_segmented,
        ae_coeff = args.ae_coeff,
        bottleneck = args.bottleneck,
        load_path = args.load_path,
        save_recon_gif = args.save_recon_gif,
        exp_name = args.gif_name,
        use_decoder = args.use_decoder,
        stochasticity = args.stochasticity,
        vision_mode = args.vision_mode
    )

if __name__ == '__main__':
    main()
