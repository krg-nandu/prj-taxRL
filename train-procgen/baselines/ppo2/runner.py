#import torch
import numpy as np
from baselines.common.runners import AbstractEnvRunner
import matplotlib.pyplot as plt
import os

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, seg_net=None):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.seg_net = seg_net

    def run(self, save_recon_gif=False, exp_name='ninjaAE128_low'):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        mb_recon = []

        # For n in range number of steps
        for _ in range(self.nsteps):
            # pass obs through the gammanet
            if type(self.seg_net) != type(None):
                X = self.obs.transpose((0, 3, 1, 2)) / 255.
                Y = torch.as_tensor(X.copy()).float().contiguous().to('cuda:1')
                Z = self.seg_net(Y)
                Z = Z.permute(0, 2, 3, 1)
                Z = torch.argmax(Z, dim=-1, keepdims=True).detach().cpu().numpy()
                # for climber N_MAX = 5 (0-4)
                Z = ((Z/4.)*255).astype(np.uint8).repeat(3, axis=-1)
                self.obs = Z.copy()

            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs, rloss, recon = self.model.step(self.obs, S=self.states, M=self.dones)
            if save_recon_gif:
                mb_recon.append(recon.copy())
                
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    denom = (info.get('max_possible_score') - info.get('min_possible_score'))
                    if denom == 0:
                        normalized_reward = 0
                    else:
                        normalized_reward = (maybeepinfo['r'] - info.get('min_possible_score')) / denom
                    maybeepinfo['nr'] = normalized_reward
                    epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        if save_recon_gif:
            mb_recon = np.asarray(mb_recon)
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            #ax2 = fig.add_subplot(122)

            if not os.path.exists(os.path.join('figures', exp_name)):
                os.mkdir(os.path.join('figures', exp_name))

            for ts in range(recon.shape[0]):
                ax1.clear()
                #ax2.clear()

                #ax1.imshow(mb_obs[ts][0])
                #ax1.axis('off')
                ax1.imshow(mb_recon[ts][0])
                ax1.axis('off')
                plt.savefig(os.path.join('figures', exp_name, 'img_%03d.png'%ts))
                plt.pause(0.001)

            # just the first will do
            os._exit(0)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


