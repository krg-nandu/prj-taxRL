GAME="heist"
LRS="0 1e0 1e2 1e3"
DIMS="64 128 256"
for lr in $LRS
do
  for dm in $DIMS
  do
    myvar=$GAME"AE"$dm"_"$lr
    CUDA_VISIBLE_DEVICES=6 python train-procgen/train_procgen/train_custom.py --env_name $GAME --distribution_mode hard --num_levels 500 --num_envs 256 --timesteps_per_proc 200_000_000 --num_episodes 1 --experiment_name blahblah --bottleneck $dm --ae_coeff 0. --load_path /media/data_cifs/projects/prj_procgen/lax/procgen/procgen/train-procgen/train_procgen/results/$GAME/$myvar/checkpoints/03050 --save_recon_gif --gif_name $myvar
    convert -resize 768x576 -delay 20 -loop 0 figures/$myvar/*.png figures/$myvar.gif
  done
done

