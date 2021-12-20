import argparse, os
import tensorflow as tf
from gym3 import types_np, VideoRecorderWrapper
from PIL import Image

from procgen import ProcgenGym3Env
from inference import load_model

def sample_gameplay(env_name, save_dir, num_levels, num_episodes, distribution_mode, vision_mode, model_path):
    prefix = 'test-ep-' if num_levels == 0 else 'train-ep-'
    video_savedir = save_dir + '/videos'
    image_savedir = save_dir + '/images'
    os.makedirs(video_savedir, exist_ok=True)
    os.makedirs(image_savedir, exist_ok=True)
    
    env = ProcgenGym3Env(
        num=1, env_name=env_name, render_mode="rgb_array",
        distribution_mode=distribution_mode, num_levels=num_levels,
        rand_seed=0, vision_mode=vision_mode
    )
    env = VideoRecorderWrapper(env=env, directory=video_savedir, prefix=prefix, info_key="rgb")

    print("loading {}...".format(model_path))
    with tf.Session(graph=tf.Graph()):
        model = load_model(model_path, env_name, num_envs=1)

        print('now recording gameplay videos of {} for {} episodes to \'{}\'...'.format(env_name, num_episodes, video_savedir))
        total_episodes, episode_reward, steps, max_steps = 0, 0, 0, 0
        image_path, video_path = None, None
        while total_episodes < num_episodes + 1:
            #env.act(types_np.sample(env.ac_space, bshape=(env.num,))) # random action
            rew, obs, first = env.observe()
            episode_reward += rew[0]
            actions, values, states, neglogpacs = model.step(obs['rgb'])
            env.act(actions)
            steps += 1

            if first:
                # save an image of the first frame
                im = Image.fromarray(obs['rgb'][0])
                image_path = '{}/{}{}.png'.format(image_savedir, prefix, f'{total_episodes:05d}')
                im.save(image_path)
                video_path = '{}/{}{}.mp4'.format(video_savedir, prefix, f'{total_episodes:05d}')
                if total_episodes != 0:
                    if total_episodes == 1: print('')
                    print('episode {} of {} ended after {} steps with a total reward of {}.'.format(total_episodes - 1, num_episodes - 1, steps, episode_reward))
                total_episodes += 1
                episode_reward = 0
                max_steps = max(max_steps, steps)
                steps = 0
        
        # clean up leftover junk (artifact of the VideoRecorderWrapper class)
        os.remove(image_path)
        os.remove(video_path)
    
    return max_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--num_levels', type=int, required=True)
    parser.add_argument('--num_episodes', type=int, required=True)
    parser.add_argument('--distribution_mode', type=str, required=True)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER" ]= "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    steps = sample_gameplay(args.env_name, args.experiment_name, args.checkpoint, args.num_levels, args.num_episodes, args.distribution_mode)
    
    #min_steps = 1000
    #while steps < min_steps:
    #    print('\n\n\n\n\ntook {} steps. trying again...\n\n\n\n\n'.format(steps))
    #    steps = sample_gameplay(args.env_name, args.experiment_name, args.checkpoint, args.num_levels, args.num_episodes, args.distribution_mode)
