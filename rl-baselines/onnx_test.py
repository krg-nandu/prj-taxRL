import os, sys
import matplotlib.pyplot as plt

sys.path.insert(0, '/media/data_cifs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main')
sys.path.insert(0, '../train-procgen/')
sys.path.insert(0, '../')
os.environ['PT_OUTPUT_DIR'] = 'testop'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import numpy as np
from collections import deque
import hyperparams as hps
from idaac_test import evaluate
from procgen import ProcgenEnv

from baselines import logger
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)

from ppo_daac_idaac import algo, utils
from ppo_daac_idaac.arguments import parser
from ppo_daac_idaac.model import PPOnet, PPOImpalaNet
from ppo_daac_idaac.storage import RolloutStorage
from ppo_daac_idaac.envs import VecPyTorchProcgen

from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
import mmcv
import tqdm, time
import onnxruntime as onnxrt


torch.backends.cudnn.benchmark = True
'''
config = '/media/data_cifs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/configs/config_test_lax.py'
checkpoint = '/media/data_cifs_lrs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/output_trainall2k1/iter_18000.pth' 
'''
config = '/media/data_cifs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/configs/config_test_x64.py'
checkpoint = '/media/data_cifs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/output_trainall2k4_64/iter_18000.pth' 


#'/media/data_cifs_lrs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/output_trainbigfishcoinrun4/iter_14000.pth' 
cfg = mmcv.Config.fromfile(config)

'''
# build the model and load checkpoint
segModel = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
#print("model={}".format(segModel))
checkpoint = load_checkpoint(segModel, checkpoint, map_location='cpu')
segModel = segModel.to('cuda:0')
segModel = segModel.eval()
'''

mean = torch.tensor(cfg['img_norm_cfg']['mean'], dtype=torch.float32).reshape(3,1,1)
std = torch.tensor(cfg['img_norm_cfg']['std'], dtype=torch.float32).reshape(3,1,1)

def extract_obs_tensor(envs):
    info = envs.venv.venv.venv.env.get_info()
    obs = np.stack([x['rgb'] for x in info])
    _obs = torch.tensor(obs, dtype=torch.float32).permute(0,3,1,2).unsqueeze(0)
    _obs = (_obs - mean) / std
    return _obs

def train(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.use_best_hps:
        args.value_epoch = hps.value_epoch[args.env_name]
        args.value_freq = hps.value_freq[args.env_name]
        args.adv_loss_coef = hps.adv_loss_coef[args.env_name]
        args.clf_hidden_size = hps.clf_hidden_size[args.env_name]
        args.order_loss_coef = hps.order_loss_coef[args.env_name]
        if args.env_name in hps.nonlin_envs:
            args.use_nonlinear_clf = True
        else:
            args.use_nonlinear_clf = False
    print("\nArguments: ", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    abspath = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(abspath)
    log_dir = os.path.join(
                    current_dir_path,
                    'results',
                    args.env_name,
                    args.experiment_name
                    )

    #log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    log_file = '-{}-{}-s{}'.format(args.env_name, args.algo, args.seed)
    logger.configure(dir=log_dir, format_strs=['csv', 'stdout'], log_suffix=log_file)
    print("\nLog File: ", log_file)

    venv = ProcgenEnv(
            num_envs=args.num_processes, 
            env_name=args.env_name,
            num_levels=args.num_levels, 
            start_level=args.start_level,
            distribution_mode=args.distribution_mode,
            stochasticity=args.stochasticity,
            vision_mode=args.vision_mode,
            render_mode='rgb_array'
    )

    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    envs = VecPyTorchProcgen(venv, device)

    if args.seg:
        obs_shape = (16,64,64)
    else:
        obs_shape = envs.observation_space.shape     
 
    actor_critic = PPOImpalaNet(
                obs_shape,
                envs.action_space.n,
                base_kwargs={'hidden_size': args.hidden_size},
                use_seg_front_end = args.seg
                )    
    actor_critic.to(device)
    print("\n Actor-Critic Network: ", actor_critic)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space)
    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)
    agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes 

    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
    ]
    onnx_session= onnxrt.InferenceSession("resnetfpn_global_b64_x64.onnx", providers=['CUDAExecutionProvider']) #providers=providers)
    #import ipdb; ipdb.set_trace()

    nsteps = torch.zeros(args.num_processes)
    for j in range(num_updates):
        actor_critic.train()

        for step in tqdm.tqdm(range(args.num_steps)):
            # Sample actions
            with torch.no_grad():
                if args.seg:
                    #import ipdb; ipdb.set_trace()
                    #_obs = extract_obs_tensor(envs)
                    #_obs = _obs.to(device)
                    #_seg = segModel.infer(_obs).squeeze()
                    #onnx_op = onnx_session.run(None, {'img': _obs.numpy()})

                    myobs = (obs * 255.).unsqueeze(0).cpu().numpy()
                    mn = mean.unsqueeze(0).unsqueeze(0).numpy()
                    st = std.unsqueeze(0).unsqueeze(0).numpy()
                    myobs = (myobs - mn)/st
                    onnx_op = onnx_session.run(None, {'img': myobs})

                    ops = torch.tensor(onnx_op[0]).squeeze().to(device)

                    im = np.vstack(ops[0].cpu().numpy())
                    plt.subplot(121)
                    #plt.imshow(_obs[0,0].permute(1,2,0).numpy())
                    plt.imshow(myobs[0,0].transpose(1,2,0))
                    plt.subplot(122)
                    plt.imshow(im)

                    plt.show()
                    value, action, action_log_prob = actor_critic.act(ops)
                else:
                    value, action, action_log_prob = actor_critic.act(rollouts.obs[step])

            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            nsteps += 1 
            nsteps[done == True] = 0
            
            rollouts.insert(obs, action, action_log_prob, value, \
                                reward, masks)

        with torch.no_grad():
            if args.seg:
                _obs = extract_obs_tensor(envs)
                _obs = _obs.to(device)
                _seg = segModel.infer(_obs).squeeze()
                next_value = actor_critic.get_value(_seg).detach()
            else:
                next_value = actor_critic.get_value(rollouts.obs[-1]).detach()
        
        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)    
        rollouts.after_update()

        # Save Model
        #if j == num_updates - 1 and args.save_dir != "":
        if (j % 10 == 0)  and args.save_dir != "":
            try:
                os.makedirs(args.save_dir)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(envs, 'ob_rms', None)
            ], os.path.join(args.save_dir, "agent{}.pt".format(log_file))) 

        # Save Logs
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps 
            print("\nUpdate {}, step {}:".format(j, total_num_steps))
            print("Last {} training episodes, mean/median reward {:.2f}/{:.2f}"\
                .format(len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards)))

            # Log training stats
            logger.logkv("train/total_num_steps", total_num_steps)            
            logger.logkv("train/mean_episode_reward", np.mean(episode_rewards))
            logger.logkv("train/median_episode_reward", np.median(episode_rewards))

            # Log eval stats (on the full distribution of levels) 
            eval_episode_rewards = evaluate(args, actor_critic, device)
            logger.logkv("test/mean_episode_reward", np.mean(eval_episode_rewards))
            logger.logkv("test/median_episode_reward", np.median(eval_episode_rewards))

            logger.dumpkvs()


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
