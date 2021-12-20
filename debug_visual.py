"""
Example random agent script using the gym3 API to demonstrate that procgen works
"""

from gym3 import types_np
from gym3 import VideoRecorderWrapper
from procgen import ProcgenGym3Env
import matplotlib.pyplot as plt

env = ProcgenGym3Env(num=1, env_name="heistH", render_mode="rgb_array")
env = VideoRecorderWrapper(env=env, directory=".", info_key="rgb")
step = 0
while True:
    env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
    rew, obs, first = env.observe()
    print(f"step {step} reward {rew} first {first}")
    if step > 0 and first:
        break
    step += 1
