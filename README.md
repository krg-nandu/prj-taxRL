# Procgen segmentation masks

## Setup

Get the code and switch to the correct branch:
```
git clone https://github.com/max-reuter/procgen.git
cd procgen
git checkout segmentation-masks
```

Set up a Conda environment:
```
conda env update --name procgen --file environment.yml
conda activate procgen
pip install -e .
# this should say "building procgen...done"
python -c "from procgen import ProcgenGym3Env; ProcgenGym3Env(num=1, env_name='coinrun')"
```

## Train a model

`--experiment_name`: a name used to ID your experiment for logging purposes.

`--vision_mode`: the visual appearance mode of the game. Current options are `normal`, `semantic_mask`, and `fg_mask`.

`--log_dir_root`: the location to log model weights and performance to. In our current experiments, we use:

```
/media/data_cifs/projects/prj_procgen/main/procgen/train-procgen/train_procgen/results/new_experiments
```
Then, progress is logged to `log_dir_root/<game>/<experiment_name>`.

### Launch training:
```
python -m train_procgen.train --game <game> --vision_mode <vision_mode> --experiment_name <experiment_name> --log_dir_root <log_dir_root>
```
