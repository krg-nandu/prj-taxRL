GAMES="climber coinrun dodgeball"
for GM in $GAMES
do
  CUDA_VISIBLE_DEVICES=0 python eval_model.py --eval_env_name $GM --env_name $GM"p" --experiment_name $GM"_p75" --use_decoder --bottleneck 256 --BASE_PATH /media/data_cifs/projects/prj_procgen/lax/procgen/procgen/train-procgen/train_procgen/results
done
