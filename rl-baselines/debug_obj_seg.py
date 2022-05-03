import os
import sys
sys.path.insert(0, '/media/data_cifs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main')
sys.path.insert(0, '../train-procgen/')
sys.path.insert(0, '../')

os.environ['PT_OUTPUT_DIR'] = 'testop'

import matplotlib.pyplot as plt
import numpy as np

from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
import mmcv

from procgen import ProcgenEnv, ProcgenGym3Env
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from gym3 import types_np, VideoRecorderWrapper

from ppo_daac_idaac.envs import VecPyTorchProcgen
import torch
from gym3.internal.renderer import Renderer

config = '/media/data_cifs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/configs/config_test_lax.py'
checkpoint = '/media/data_cifs_lrs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/output_trainbigfishcoinrun4/iter_14000.pth' 
cfg = mmcv.Config.fromfile(config)

# build the model and load checkpoint
model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
print("model={}".format(model))
checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

print('**')

venv = ProcgenEnv(
            num_envs=1, 
            env_name='bigfish',
            num_levels=0, 
            start_level=0, 
            distribution_mode='hard',
            vision_mode='normal',
            stochasticity=1.,
            render_mode='rgb_array')
venv = VecExtractDictObs(venv, "rgb")
venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
venv = VecNormalize(venv=venv, ob=False)

obs = venv.reset()
info = venv.venv.venv.venv.env.get_info()

#eval_envs = VecPyTorchProcgen(env, 'cpu')
#obs = eval_envs.reset()

with torch.no_grad():
    im = torch.tensor(info[0]['rgb'], dtype=torch.float32).permute(2,0,1).unsqueeze(0).unsqueeze(0)
    result = model(im, img_metas=None, return_loss=False)
