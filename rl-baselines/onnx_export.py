import os, sys
import matplotlib.pyplot as plt

sys.path.insert(0, '/media/data_cifs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main')
sys.path.insert(0, '../train-procgen/')
sys.path.insert(0, '../')
os.environ['PT_OUTPUT_DIR'] = 'testop'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import numpy as np
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
import mmcv
import tqdm, time

torch.backends.cudnn.benchmark = True

'''
config = '/media/data_cifs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/configs/config_test_lax.py'
checkpoint = '/media/data_cifs_lrs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/output_trainall2k1/iter_18000.pth' 
#checkpoint = '/media/data_cifs_lrs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/output_trainbigfish4/iter_2500.pth'

config = '/media/data_cifs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/configs/config_test_x64.py'
checkpoint = '/media/data_cifs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/output_trainall2k4_64/iter_18000.pth' 

'''

config = '/media/data_cifs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/configs/config_test_x128LW.py'
checkpoint = '/media/data_cifs_lrs/projects/prj_rl/alekh/The-Emergence-of-Objectness-main/output_trainbigfish10_128multiscaleboth_LW/iter_13000.pth' 

cfg = mmcv.Config.fromfile(config)

# build the model and load checkpoint
segModel = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(segModel, checkpoint, map_location='cpu')
segModel = segModel.eval()

dummy_input = torch.randn(1, 32, 3, 128, 128)
input_names = ["img"]
output_names = ["all_pred_mask"]

torch.onnx.export(segModel, 
                  dummy_input,
                  "RFPN_MultiScale_b32_x128LW.onnx",
                  opset_version=11,
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )
