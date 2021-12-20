import matplotlib.pyplot as plt
from constants import ENV_NAMES

FIRST_VALID = 30

if __name__ == '__main__':
    experiments = {'normal': 'r', 'semantic_mask': 'k', 'fg_mask': 'b'}
    root = "/media/data_cifs/projects/prj_procgen/main/procgen/train-procgen/train_procgen/results/new_experiments"
    destination = f"{root}/results"
    
    for game in ENV_NAMES:
        print(f"Plotting {game}...")
        fig = plt.figure()
        fig.suptitle(game, fontsize=16)
        ax = fig.add_subplot()
        ax.set_ylabel('normalized score', fontsize=16)
        ax.set_xlabel('training updates (200M timesteps total)', fontsize=12)
        
        for experiment_dir, color in experiments.items():
            file = f"{root}/{game}/{experiment_dir}/progress.csv"
            try:
                trace = open(file).readlines()[FIRST_VALID:]
                epnormrewmeans = [float(y.split(',')[1]) for y in trace]
                ax.plot(epnormrewmeans, c=color, alpha=0.5, label=experiment_dir, linewidth=0.4)
            except FileNotFoundError:
                continue

        ax.set_ylim(bottom=0, top=1.1)
        legend = plt.legend()
        for legobj in legend.legendHandles:
            legobj.set_linewidth(3.0)
        plt.show()

        savepath = f"{destination}/{game}.png"
        plt.savefig(savepath, dpi=500)
