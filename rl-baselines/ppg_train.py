# -*- coding: utf-8 -*-

import os, sys
sys.path.insert(0, '../train-procgen/')
sys.path.insert(0, '../')

import argparse
from mpi4py import MPI
from ppg import ppg
from ppg import torch_util as tu
from ppg.impala_cnn import ImpalaEncoder
from ppg import logger
from ppg.envs import get_venv

def train_fn(
    env_name="coinrun",
    distribution_mode="hard",
    arch="dual",  # 'shared', 'detach', or 'dual'
    # 'shared' = shared policy and value networks
    # 'dual' = separate policy and value networks
    # 'detach' = shared policy and value networks, but with the value function gradient detached during the policy phase to avoid interference
    interacts_total=100000000,
    num_envs=64,
    n_epoch_pi=1,
    n_epoch_vf=1,
    gamma=.999,
    aux_lr=5e-4,
    lr=5e-4,
    nminibatch=8,
    aux_mbsize=4,
    clip_param=.2,
    kl_penalty=0.0,
    n_aux_epochs=6,
    n_pi=32,
    beta_clone=1.0,
    vf_true_weight=1.0,
    comm=None,
    experiment_name=None,
    num_levels=None,
    start_level=None,
    vision_mode=None,
    stochasticity=None,
    port=44000
    ):

    if comm is None:
        comm = MPI.COMM_WORLD
    tu.setup_dist(comm=comm, start_port=port)
    tu.register_distributions_for_tree_util()

    abspath = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(abspath)
    log_dir = os.path.join(
                    current_dir_path,
                    'results',
                    env_name,
                    experiment_name
                    )

    if os.path.isdir(log_dir):
        print('Error: {}/ is already a directory, try a different --experiment_name.'.format(log_dir))
        os._exit(0)
    else:
        os.makedirs(log_dir)
        print('logging results to {}/.'.format(log_dir))


    if log_dir is not None:
        format_strs = ['csv', 'stdout'] if comm.Get_rank() == 0 else []
        logger.configure(comm=comm, dir=log_dir, format_strs=format_strs)
    venv = get_venv(
            num_envs=num_envs, 
            env_name=env_name, 
            distribution_mode=distribution_mode, 
            vision_mode=vision_mode, 
            stochasticity=stochasticity,
            num_levels=num_levels,
            start_level=start_level
            )

    enc_fn = lambda obtype: ImpalaEncoder(
        obtype.shape,
        outsize=256,
        chans=(16, 32, 32),
    )
    model = ppg.PhasicValueModel(venv.ob_space, venv.ac_space, enc_fn, arch=arch)

    model.to(tu.dev())
    logger.log(tu.format_model(model))
    tu.sync_params(model.parameters())

    name2coef = {"pol_distance": beta_clone, "vf_true": vf_true_weight}

    ppg.learn(
        venv=venv,
        model=model,
        interacts_total=interacts_total,
        ppo_hps=dict(
            lr=lr,
            γ=gamma, 
            λ=0.95,
            nminibatch=nminibatch,
            n_epoch_vf=n_epoch_vf,
            n_epoch_pi=n_epoch_pi,
            clip_param=clip_param,
            kl_penalty=kl_penalty,
            log_save_opts={"save_mode": "last"}
        ),
        aux_lr=aux_lr,
        aux_mbsize=aux_mbsize,
        n_aux_epochs=n_aux_epochs,
        n_pi=n_pi,
        name2coef=name2coef,
        comm=comm,
    )

def main():
    parser = argparse.ArgumentParser(description='Process PPG training arguments.')

    ## ENV HYPERPARAMETERS
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"], required=True)
    parser.add_argument('--num_levels', type=int, required=True)
    parser.add_argument('--start_level', type=int, default=0)

    parser.add_argument('--test_worker_interval', type=int, default=0)

    parser.add_argument('--timesteps_per_proc', type=int, default=200_000_000)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=0)
    parser.add_argument('--test_eval', type=bool, default=False)


    parser.add_argument('--vision_mode', type=str, default='normal', choices=["normal", "semantic_mask", "fg_mask"], required=True)
    parser.add_argument('--stochasticity', type=float, default=1.)

    ## PPG HYPERPARAMETERS
    parser.add_argument('--port', type=int, default=44000, required=True)
    parser.add_argument('--n_epoch_pi', type=int, default=1)
    parser.add_argument('--n_epoch_vf', type=int, default=1)
    parser.add_argument('--n_aux_epochs', type=int, default=6)
    parser.add_argument('--n_pi', type=int, default=32)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--kl_penalty', type=float, default=0.0)
    parser.add_argument('--arch', type=str, default='dual') # 'shared', 'detach', or 'dual'



    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    train_fn(
        env_name=args.env_name,
        num_envs=args.num_envs,
        distribution_mode=args.distribution_mode,
        interacts_total=args.timesteps_per_proc,
        vision_mode=args.vision_mode,
        stochasticity=args.stochasticity,
        n_epoch_pi=args.n_epoch_pi,
        n_epoch_vf=args.n_epoch_vf,
        n_aux_epochs=args.n_aux_epochs,
        n_pi=args.n_pi,
        arch=args.arch,
        comm=comm,
        experiment_name=args.experiment_name,
        num_levels=args.num_levels,
        start_level=args.start_level,
        port=args.port
    )

if __name__ == '__main__':
    main()
